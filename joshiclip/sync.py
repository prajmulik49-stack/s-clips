"""A/V sync and speed checks of a rendered Short against the normalized source span.

For probe points at the start / middle / end of the output:
  audio lag = argmax cross-correlation of 5 ms log-RMS envelopes (output vs source), sub-sample refined
  video lag = argmax cross-correlation of frame-activity (mean |frame - prev|) at the output fps
  A/V offset = video lag - audio lag  (0 = lips match the voice exactly as in the source)
A video lag that grows from start to end = speed error (slow-mo / fast-mo).
"""
from __future__ import annotations

import numpy as np

from . import media


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag L (samples, sub-sample refined) such that a[i] ~ b[i + max_lag + L]; b is a padded window around a."""
    a = _z(a)
    cs = []
    for L in range(0, 2 * max_lag + 1):
        bb = b[L:L + len(a)]
        if len(bb) < len(a):
            cs.append(-1e9); continue
        v = float(np.dot(a, _z(bb)) / len(a)) if len(a) else -1e9
        cs.append(v if np.isfinite(v) else -1e9)
    cs = np.array(cs)
    k = int(np.argmax(cs))
    frac = 0.0
    if 0 < k < len(cs) - 1:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return k - max_lag + frac, float(cs[k])


def _env(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return np.log(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, 1)) + 1e-4)


def audio_lag_ms(out_path, src_path, t_out: float, t_src: float, L: float, search: float = 0.25, sr: int = 16000) -> tuple[float, float]:
    hop = sr // 200  # 5 ms
    o = _env(media.read_audio(out_path, sr, t0=t_out, dur=L), hop)
    s = _env(media.read_audio(src_path, sr, t0=t_src - search, dur=L + 2 * search), hop)
    lag, c = xcorr_lag(o, s, int(search * 200))
    return lag * 5.0, c


def video_lag_ms(out_path, src_path, t_out: float, t_src: float, L: float, fps: float, search: float = 0.25) -> tuple[float, float]:
    o = media.read_gray(out_path, t_out, L, fps)
    s = media.read_gray(src_path, t_src - search, L + 2 * search, fps)
    oa = np.abs(np.diff(o, axis=0)).mean((1, 2))
    sa = np.abs(np.diff(s, axis=0)).mean((1, 2))
    lag, c = xcorr_lag(oa, sa, int(round(search * fps)))
    return lag * 1000.0 / fps, c


def _strips(img: np.ndarray, frac: float = 0.15) -> np.ndarray:
    w = img.shape[-1]
    k = max(2, int(w * frac))
    return np.concatenate([img[..., :k], img[..., -k:]], axis=-1)


def video_lag_vertical_ms(out_path, src_path, t_out: float, t_src: float, L: float, fps: float, vgeom: dict,
                          search: float = 0.25) -> tuple[float, float]:
    """Vertical renders: rebuild the expected footage from the source with the SAME camera path, compare only the
    outer left/right strips of the footage area (no sticker / captions there)."""
    W, H, vy, vh = vgeom["W"], vgeom["H"], vgeom["video_y"], vgeom["video_h"]
    sw, sh = 480, 270
    o = media.read_gray(out_path, t_out, L, fps, W // 4, H // 4)[:, vy // 4:(vy + vh) // 4]
    s = media.read_gray(src_path, t_src - search, L + 2 * search, fps, sw, sh)
    cx, z, f0 = np.asarray(vgeom["cx"]), np.asarray(vgeom["zoom"]), vgeom["frame0"]
    import cv2
    oh, ow = o.shape[1:]
    rows = []
    for j in range(len(s)):
        k = int(np.clip(round((t_src - search) * fps) + j - f0, 0, len(cx) - 1))
        ch = sh / z[k]; cw = ch * W / vh
        x0 = float(np.clip(cx[k] * sw - cw / 2, 0, sw - cw)); y0 = (sh - ch) / 2
        M = np.array([[cw / ow, 0, x0], [0, ch / oh, y0]])
        rows.append(cv2.warpAffine(s[j], M, (ow, oh), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP))
    s2 = np.array(rows)
    oa = np.abs(np.diff(_strips(o), axis=0)).mean((1, 2))
    sa = np.abs(np.diff(_strips(s2), axis=0)).mean((1, 2))
    lag, c = xcorr_lag(oa, sa, int(round(search * fps)))
    return lag * 1000.0 / fps, c


def check_render(out_path, src_path, span: tuple[float, float], fps: float, probe_len: float = 1.5,
                 max_offset_ms: float = 40.0, vgeom: dict | None = None) -> dict:
    """Duration / fps / A/V offset checks for one continuous-span render."""
    a, b = span
    info = media.probe(out_path)
    v, au = info["video"], info["audio"]
    nf = media.count_frames(out_path)
    exp_frames = int(round((b - a) * fps))
    res = {"out_fps": v["avg_fps"], "expected_fps": fps, "frames": nf, "expected_frames": exp_frames,
           "video_dur": v["duration"], "audio_dur": au["duration"], "span_s": round(b - a, 6), "probes": []}
    L = min(probe_len, (b - a) / 3)
    for where, t in (("start", 0.0), ("mid", (b - a - L) / 2), ("end", b - a - L)):
        al, ac = audio_lag_ms(out_path, src_path, t, a + t, L)
        if vgeom:
            vl, vc = video_lag_vertical_ms(out_path, src_path, t, a + t, L, fps, vgeom)
        else:
            vl, vc = video_lag_ms(out_path, src_path, t, a + t, L, fps)
        res["probes"].append({"where": where, "t_out": round(t, 3), "audio_lag_ms": round(al, 1), "audio_corr": round(ac, 3),
                              "video_lag_ms": round(vl, 1), "video_corr": round(vc, 3), "av_offset_ms": round(vl - al, 1)})
    rel = [p for p in res["probes"] if p["audio_corr"] > 0.5 and p["video_corr"] > 0.3]
    res["n_reliable_probes"] = len(rel)
    res["max_abs_av_offset_ms"] = round(max([abs(p["av_offset_ms"]) for p in rel] or [0.0]), 1)
    res["max_abs_audio_lag_ms"] = round(max(abs(p["audio_lag_ms"]) for p in res["probes"] if p["audio_corr"] > 0.5) if any(p["audio_corr"] > 0.5 for p in res["probes"]) else 0.0, 1)
    n_audio_ok = sum(1 for p in res["probes"] if p["audio_corr"] > 0.5)
    res["checks"] = {
        "probes_reliable": len(rel) >= 2 and n_audio_ok >= 2,
        "fps_match": abs(v["avg_fps"] - fps) < 1e-3,
        "frames_match": abs(nf - exp_frames) <= 1,
        "audio_len_match": abs(au["duration"] - (b - a)) <= 1.0 / fps + 0.025,  # AAC frame granularity
        "av_offset_ok": res["max_abs_av_offset_ms"] <= max_offset_ms,
        "audio_aligned": res["max_abs_audio_lag_ms"] <= max_offset_ms,
    }
    res["checks"] = {k: bool(v) for k, v in res["checks"].items()}
    res["ok"] = all(res["checks"].values())
    return res
