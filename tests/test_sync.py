"""Step 1: normalization + frame-exact cutting keep lip-sync and speed (synthetic flash/beep ground truth)."""
from __future__ import annotations

import numpy as np
import pytest

from joshiclip import config, cut, ingest, media, sync

EVENTS = [0.5, 1.3, 2.1, 2.75, 3.6, 4.45, 5.2, 6.05, 6.9, 7.6]   # flash + beep at the same instant
DUR = 8.5


def _make_source(path, fps_expr: str, rate: int, vfr_jitter: float, audio_sr: int):
    en = "+".join(f"between(t,{t:.3f},{t + 0.05:.3f})" for t in EVENTS)
    vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{en}'"
    if vfr_jitter:
        vf += f",setpts='(N+{vfr_jitter}*sin(N*1.7))/({fps_expr})/TB'"
    media.run([media.ff(), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s=320x180:r={fps_expr}:d={DUR}",
               "-f", "lavfi", "-i", f"aevalsrc='0.8*sin(2*PI*1000*t)*({en})':s={audio_sr}:d={DUR}",
               "-vf", vf, "-fps_mode", "vfr" if vfr_jitter else "cfr", "-c:v", "libx264", "-preset", "ultrafast",
               "-c:a", "aac", "-b:a", "128k", str(path)])


def _flash_times(path, fps):
    g = media.read_gray(path, 0, 60, fps, 32, 18).mean((1, 2))
    on = g > 128
    return [i / fps for i in range(1, len(on)) if on[i] and not on[i - 1]] + ([0.0] if on[0] else [])


def _beep_times(path):
    sr = 48000
    x = np.abs(media.read_audio(path, sr))
    hop = 48
    env = x[: len(x) // hop * hop].reshape(-1, hop).max(1)
    on = env > 0.3
    return [i * hop / sr for i in range(1, len(on)) if on[i] and not on[i - 1]]


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.mark.parametrize("fps_expr,target,jitter", [("60", 60, 0.35), ("30000/1001", 30, 0.3), ("30", 30, 0.0)])
def test_normalize_and_cut_keep_sync(tmp_path, cfg, fps_expr, target, jitter):
    src = tmp_path / "src.mkv"
    _make_source(src, fps_expr, 60 if target == 60 else 30, jitter, 44100)
    meta = ingest.normalize(cfg, src, tmp_path / "w")
    assert meta["fps"] == target
    assert meta["checks"]["ok"]
    norm = tmp_path / "w" / "norm.mp4"

    # events in norm.mp4 sit exactly where they are in the source (no drift, no offset added)
    src_fl, src_bp = _flash_times(src, target), _beep_times(src)
    fl, bp = _flash_times(norm, target), _beep_times(norm)
    assert len(fl) == len(EVENTS) and len(bp) == len(EVENTS)
    for sf, sb, f, b in zip(src_fl, src_bp, fl, bp):
        assert abs(f - sf) <= 1.0 / target + 1e-6, (sf, f)
        assert abs(b - sb) <= 0.003, (sb, b)

    # cut a span that starts/ends at odd times; must be frame-snapped, exact length, in sync
    span = cut.snap(1.93, 6.61, target)
    out = tmp_path / "cut.mp4"
    media.run(cut.cut_plain(norm, span, out, cfg["render"]["draft"]))
    assert media.count_frames(out) == span.fb - span.fa
    assert abs(media.probe(out)["video"]["avg_fps"] - target) < 1e-3
    keep = [i for i, b in enumerate(bp) if span.start + 0.02 < b < span.end - 0.08]
    fo, bo = _flash_times(out, target), _beep_times(out)
    assert len(fo) == len(keep) and len(bo) == len(keep)
    for i, f, b in zip(keep, fo, bo):
        assert abs((f + span.start) - fl[i]) <= 1.0 / target + 1e-6, "video moved (speed/offset error)"
        assert abs((b + span.start) - bp[i]) <= 0.005, "audio moved (offset error)"
        assert abs((f - b) - (fl[i] - bp[i])) * 1000 <= cfg["sync"]["max_av_offset_ms"]

    # the generic cross-correlation checker agrees
    rep = sync.check_render(out, norm, (span.start, span.end), target, probe_len=1.2)
    assert rep["checks"]["fps_match"] and rep["checks"]["frames_match"] and rep["checks"]["audio_len_match"]
    assert rep["max_abs_audio_lag_ms"] <= 40


def test_snap_integer_samples():
    for fps in (30, 60):
        s = cut.snap(12.3456, 57.891, fps)
        assert s.n_samples == (s.fb - s.fa) * 48000 // fps
        assert (s.n_samples * fps) % 48000 == 0
        assert abs(s.start - 12.3456) <= 0.5 / fps and abs(s.end - 57.891) <= 0.5 / fps
    with pytest.raises(ValueError):
        cut.snap(0, 1, 7)


def test_nominal_fps():
    mk = lambda f: {"video": {"avg_fps": f, "fps": f}}
    assert ingest.nominal_fps(mk(29.97), [30, 60]) == 30
    assert ingest.nominal_fps(mk(59.94), [30, 60]) == 60
    assert ingest.nominal_fps(mk(47.3), [30, 60]) == 60
    assert ingest.nominal_fps(mk(24.0), [30, 60]) == 30
