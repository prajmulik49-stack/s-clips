"""Quiet background music under the original sound.

- one track per Short, picked from music/ by a seeded random choice (same Short -> same track on every re-render)
- random start point inside the track, looped if the Short is longer, 0.6 s fade in / 1.5 s fade out
- level is set RELATIVE to the Short's own loudness: `gap_db` below it when nobody talks, `speech_db` below it under speech
- EQ keeps it out of the voice: high-pass 120 Hz + a wide dip around 2.5 kHz (where words are understood)
The returned buffer has exactly the input length; the voice itself is never touched.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np

EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus")


def tracks(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS and p.is_file())


def pick(folder: Path, seed: str) -> Path | None:
    ts = tracks(folder)
    if not ts:
        return None
    r = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    return ts[r.randrange(len(ts))]


def _decode(path: Path, sr: int) -> np.ndarray:
    from . import media
    return media.read_audio(path, sr, mono=False).astype(np.float32)


def bed(path: Path, n: int, sr: int, seed: str, eq_af: str, ff_filter) -> np.ndarray:
    """n samples (stereo) of the track: random start, looped, EQ'd, faded."""
    x = _decode(path, sr)
    if len(x) < sr:
        return np.zeros((n, 2), np.float32)
    r = random.Random(int(hashlib.sha256(("start" + seed).encode()).hexdigest()[:12], 16))
    # skip the first/last 10 % (intros and endings sound like intros and endings)
    lo, hi = int(len(x) * 0.1), max(int(len(x) * 0.1) + 1, int(len(x) * 0.9) - n)
    s0 = r.randrange(lo, hi) if hi > lo else 0
    out = np.zeros((n, 2), np.float32)
    k, pos = 0, s0
    xf = 0.25
    while k < n:
        chunk = x[pos:pos + (n - k)]
        if k and len(chunk) > int(xf * sr):          # short fade-in at every loop seam
            f = np.linspace(0, 1, int(xf * sr), dtype=np.float32)[:, None]
            chunk = chunk.copy(); chunk[:len(f)] *= f
        out[k:k + len(chunk)] = chunk
        k += len(chunk)
        pos = lo                                     # loop from the 10 % point
    out = ff_filter(out, eq_af)
    fi, fo = int(0.6 * sr), int(1.5 * sr)
    out[:fi] *= np.linspace(0, 1, min(fi, n), dtype=np.float32)[:, None]
    out[n - min(fo, n):] *= np.linspace(1, 0, min(fo, n), dtype=np.float32)[:, None]
    return out


def mix(main: np.ndarray, gate: np.ndarray, cfg: dict, seed: str, sr: int, ff_filter, loudness, log=print) -> tuple[np.ndarray, dict]:
    """main: (n, 2) final-ish mix; gate: 0..1 speech presence per sample. Returns (main + music, report)."""
    from .config import ROOT
    folder = Path(cfg.get("folder", "music"))
    folder = folder if folder.is_absolute() else ROOT / folder
    tr = pick(folder, seed)
    if tr is None:
        return main, {"music": None}
    n = len(main)
    eq = (f"highpass=f={cfg.get('highpass_hz', 120)},"
          f"equalizer=f=2500:t=q:w=1.0:g={cfg.get('voice_band_cut_db', -5)},"
          f"lowpass=f={cfg.get('lowpass_hz', 12000)}")
    m = bed(tr, n, sr, seed, eq, ff_filter)
    L_main, L_m = loudness(main), loudness(m)
    if not np.isfinite(L_m) or not np.isfinite(L_main):
        return main, {"music": tr.name, "skipped": "silent"}
    base = 10 ** ((L_main + cfg.get("gap_db", -20) - L_m) / 20)            # level when nobody talks
    under = 10 ** ((cfg.get("speech_db", -28) - cfg.get("gap_db", -20)) / 20)  # extra dip under speech
    g = base * (1 + (under - 1) * gate)
    out = main + m * g[:, None]
    log(f"    music: {tr.name} ({cfg.get('gap_db', -20)} / {cfg.get('speech_db', -28)} dB under the voice)")
    return out.astype(np.float32), {"music": tr.name, "gap_db": cfg.get("gap_db", -20), "speech_db": cfg.get("speech_db", -28)}
