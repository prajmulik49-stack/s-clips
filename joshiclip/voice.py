"""Voice pipeline for one Short (48 kHz, exact length preserved and asserted at every step).

source span -> Demucs (vocals | background) -> DeepFilterNet on vocals -> optional deeper pitch (rubberband,
duration-preserving) -> warm EQ + gentle compression -> small-room reverb + light slap echo (voice only,
tails trimmed) -> background ducked under speech (attack 50 ms / release 300 ms) -> mix -> -14 LUFS, <= -1 dBTP.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import media, tools

SR = 48000


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], x.dtype)
    return np.concatenate([x, pad])


def _ff_filter(x: np.ndarray, af: str) -> np.ndarray:
    """Run a mono/stereo float buffer through an ffmpeg audio filter; length restored exactly."""
    n = len(x)
    ch = 1 if x.ndim == 1 else x.shape[1]
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "i.wav", Path(td) / "o.wav"
        sf.write(i, x, SR, subtype="FLOAT")
        media.run([media.ff(), "-v", "error", "-y", "-i", str(i), "-af", af, "-ar", str(SR), "-ac", str(ch),
                   "-c:a", "pcm_f32le", str(o)])
        y, sr = sf.read(o, dtype="float32", always_2d=False)
    assert sr == SR
    return _fit(y, n)


def separate(x: np.ndarray, threads: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Demucs htdemucs -> (vocals mono, background stereo) at 48 kHz, same length."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from scipy.signal import resample_poly
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    m = get_model("htdemucs")
    m.eval()
    st = x if x.ndim == 2 else np.stack([x, x], 1)
    y = resample_poly(st, 147, 160, axis=0).astype(np.float32)          # 48k -> 44.1k
    with torch.inference_mode():
        out = apply_model(m, torch.from_numpy(y.T.copy())[None], shifts=0, split=True, overlap=0.25, progress=False)[0]
    src = {n: out[k].numpy().T for k, n in enumerate(m.sources)}
    voc = resample_poly(src["vocals"], 160, 147, axis=0)
    bg = resample_poly(sum(src[n] for n in m.sources if n != "vocals"), 160, 147, axis=0)
    return _fit(voc.mean(1).astype(np.float32), len(x)), _fit(bg.astype(np.float32), len(x))


import threading

_DF = {}
_DF_LOCK = threading.Lock()


def denoise(v: np.ndarray) -> np.ndarray:
    """DeepFilterNet3. The model is loaded once (its logger setup is not thread-safe) and calls are serialized."""
    import torch
    from df.enhance import enhance, init_df
    with _DF_LOCK:
        if "m" not in _DF:
            torch.manual_seed(0)
            _DF["m"], _DF["s"], _ = init_df(log_level="ERROR")
        y = enhance(_DF["m"], _DF["s"], torch.from_numpy(v[None].copy())).numpy()[0]
    return _fit(y.astype(np.float32), len(v))


def room_ir(decay_s: float, predelay_s: float, highcut: float, seed: int = 7) -> np.ndarray:
    """Deterministic small-room impulse response: pre-delay + exponentially decaying filtered noise."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    n = int(SR * decay_s * 1.5)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / decay_s)                  # -60 dB at decay_s
    ir = rng.standard_normal(n) * env
    ir = sosfilt(butter(4, highcut, "low", fs=SR, output="sos"), ir)
    ir = sosfilt(butter(2, 250, "high", fs=SR, output="sos"), ir)
    ir = np.concatenate([np.zeros(int(predelay_s * SR)), ir])
    return (ir / np.sqrt((ir ** 2).sum())).astype(np.float32)


def add_space(v: np.ndarray, c: dict) -> np.ndarray:
    from scipy.signal import fftconvolve
    n = len(v)
    wet = fftconvolve(v, room_ir(c["reverb_decay_s"], c["reverb_predelay_s"], c["reverb_highcut_hz"]))[:n]
    wet *= np.sqrt((v ** 2).mean() / max(1e-12, (wet ** 2).mean()))
    out = (1 - c["reverb_wet"]) * v + c["reverb_wet"] * wet
    for d_ms, g_db in c["echo_taps"]:                  # light slap echo, trimmed to the span
        d = int(SR * d_ms / 1000)
        out[d:] += v[:n - d] * 10 ** (g_db / 20)
    return out.astype(np.float32)


def speech_gate(v: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """0..1 speech-presence envelope from the clean vocal (10 ms frames), attack/release smoothed per sample."""
    hop = SR // 100
    nf = len(v) // hop + 1
    e = np.array([np.sqrt(np.mean(v[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(nf)])
    db = 20 * np.log10(e + 1e-9)
    thr = max(np.percentile(db, 60) - 12, -50)
    on = (db > thr).astype(np.float32)
    on = np.convolve(on, np.ones(15) / 15, mode="same") > 0.2          # bridge short gaps (150 ms)
    tgt = np.repeat(on.astype(np.float32), hop)[: len(v)]
    tgt = _fit(tgt, len(v))
    a_c = np.exp(-1 / (SR * attack_ms / 1000)); r_c = np.exp(-1 / (SR * release_ms / 1000))
    from scipy.signal import lfilter
    # asymmetric one-pole: process attack (rising) and release (falling) via max of two smoothers
    up = lfilter([1 - a_c], [1, -a_c], tgt)
    down = lfilter([1 - r_c], [1, -r_c], tgt)
    return np.clip(np.maximum(up, down), 0, 1).astype(np.float32)


def true_peak_db(x: np.ndarray) -> float:
    from scipy.signal import resample_poly
    y = resample_poly(x, 4, 1, axis=0)
    return float(20 * np.log10(np.abs(y).max() + 1e-12))


def loudness(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(x))


def sbr_db(voice: np.ndarray, bg: np.ndarray) -> float:
    v = voice if voice.ndim == 1 else voice.mean(1)
    b = bg if bg.ndim == 1 else bg.mean(1)
    return float(10 * np.log10((v ** 2).mean() / max(1e-12, (b ** 2).mean())))


def process(src_wav: Path, out_wav: Path, c: dict, log=print) -> dict:
    x, sr = sf.read(src_wav, dtype="float32", always_2d=True)
    assert sr == SR
    n = len(x)
    rep = {"samples": n}
    if c.get("separator", "demucs") == "none":
        # keep the ORIGINAL sound: no separation, no denoise, no ducking. Only light warm EQ + a little echo.
        eq = (f"equalizer=f={c['low_shelf_hz']}:t=h:w=200:g={c['low_shelf_db']},"
              f"equalizer=f=350:t=q:w=1.2:g={c['mud_cut_db']},equalizer=f=4000:t=q:w=1.0:g={c['presence_db']},"
              f"acompressor=threshold=-20dB:ratio={c['comp_ratio']}:attack=8:release=120:makeup=1")
        if c["pitch_semitones"]:
            ratio = 2 ** (c["pitch_semitones"] / 12)
            eq = f"rubberband=pitch={ratio:.5f}:formant={c['formant']}:transients=smooth," + eq
        y = _ff_filter(x, eq)
        mix = np.stack([add_space(y[:, ch].copy(), c) for ch in range(y.shape[1])], 1)
        rep.update(separator="none", sbr_source_db=None, sbr_output_db=None)
        return _finish_loudness(mix, n, out_wav, c, rep)
    if c.get("separator", "demucs") == "dfn":
        # fast: DeepFilterNet on the mix = clean voice; background = mix - voice (ambience/music kept for ducking)
        voc = denoise(x.mean(1).astype(np.float32))
        bg = (x - voc[:, None]).astype(np.float32)
        v = voc
    else:
        voc, bg = separate(x)
    rep["separator"] = c.get("separator", "demucs")
    rep["sbr_source_db"] = round(sbr_db(voc, bg), 2)
    assert len(voc) == n and len(bg) == n
    if rep["separator"] != "dfn":
        v = denoise(voc) if c["denoise"] else voc
    assert len(v) == n
    if c["pitch_semitones"]:
        ratio = 2 ** (c["pitch_semitones"] / 12)
        v = _ff_filter(v, f"rubberband=pitch={ratio:.5f}:formant={c['formant']}:transients=smooth")
    eq = (f"equalizer=f={c['low_shelf_hz']}:t=h:w=200:g={c['low_shelf_db']},"
          f"equalizer=f=350:t=q:w=1.2:g={c['mud_cut_db']},equalizer=f=4000:t=q:w=1.0:g={c['presence_db']},"
          f"acompressor=threshold=-20dB:ratio={c['comp_ratio']}:attack=8:release=120:makeup=2")
    v = _ff_filter(v, eq)
    v = add_space(v, c)
    assert len(v) == n
    gate = speech_gate(voc, c["duck_attack_ms"], c["duck_release_ms"])
    duck = 10 ** (c["duck_db"] / 20)
    rest = 10 ** (c["bg_gap_db"] / 20)
    g = rest + (duck - rest) * gate
    bgd = bg * g[:, None]
    mix = v[:, None] + bgd
    rep["sbr_output_db"] = round(sbr_db(v, bgd), 2)
    rep["speech_frac"] = round(float((gate > 0.5).mean()), 3)
    return _finish_loudness(mix, n, out_wav, c, rep)


def _finish_loudness(mix: np.ndarray, n: int, out_wav: Path, c: dict, rep: dict) -> dict:
    """-14 LUFS, true peak <= -1 dBTP (static gain, then a lookahead limiter only if needed); length asserted."""
    L = loudness(mix)
    mix = mix * 10 ** ((c["target_lufs"] - L) / 20)
    if true_peak_db(mix) > c["true_peak_db"]:
        mix = _ff_filter(mix, f"alimiter=limit={10 ** ((c['true_peak_db'] - 0.6) / 20):.4f}:attack=5:release=60:level=disabled")
        L2 = loudness(mix)
        mix = mix * 10 ** ((c["target_lufs"] - L2) / 20)
        k = 0
        while true_peak_db(mix) > c["true_peak_db"] and k < 6:
            mix *= 10 ** (-0.2 / 20); k += 1
    assert len(mix) == n, (len(mix), n)
    rep["lufs"] = round(loudness(mix), 2)
    rep["true_peak_dbtp"] = round(true_peak_db(mix), 2)
    sf.write(out_wav, mix.astype(np.float32), SR, subtype="FLOAT")
    return rep


def extract_span(norm: Path, span, out_wav: Path) -> None:
    """Exact-sample span from norm.mp4 (pts-based atrim after a coarse seek)."""
    a = span.sa / span.sr; b = span.sb / span.sr
    media.run([media.ff(), "-v", "error", "-y", "-ss", f"{max(0.0, span.start - 3):.6f}", "-copyts", "-i", str(norm),
               "-vn", "-af", f"atrim=start={a:.9f}:end={b:.9f},asetpts=PTS-STARTPTS", "-ac", "2", "-ar", str(SR),
               "-c:a", "pcm_f32le", str(out_wav)])
    x, _ = sf.read(out_wav, dtype="float32", always_2d=True)
    assert abs(len(x) - span.n_samples) <= 1, (len(x), span.n_samples)
    if len(x) != span.n_samples:
        sf.write(out_wav, _fit(x, span.n_samples), SR, subtype="FLOAT")


def extract_segments(path: Path, spans: list, out_wav: Path, fade_ms: float = 0.0) -> None:
    """Exact-sample audio for several frame-exact segments, joined in one graph (same cut points as the video)."""
    from . import cut
    g = cut.multi_trim_graph(spans, aout="a", fade_ms=fade_ms, want_video=False)
    media.run([media.ff(), "-v", "error", "-y", "-ss", f"{max(0.0, spans[0].start - 3):.6f}", "-copyts", "-i", str(path),
               "-filter_complex", g, "-map", "[a]", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(out_wav)])
    x, _ = sf.read(out_wav, dtype="float32", always_2d=True)
    want = sum(s.n_samples for s in spans)
    assert abs(len(x) - want) <= len(spans), (len(x), want)
    if len(x) != want:
        sf.write(out_wav, _fit(x, want), SR, subtype="FLOAT")
