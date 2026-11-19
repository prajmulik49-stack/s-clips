"""Audio analysis on audio_16k_mono.wav: Silero VAD, RMS, pitch (YIN), YAMNet events. All deterministic, CPU."""
from __future__ import annotations

import numpy as np

SR = 16000


def silero_vad(x: np.ndarray, model_path: str) -> np.ndarray:
    """Speech probability per 32 ms chunk (Silero v5 ONNX, context-64 wrapper as in the reference implementation)."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    n = len(x) // 512
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, 64), np.float32)
    sr = np.array(SR, np.int64)
    out = np.zeros(n, np.float32)
    for i in range(n):
        chunk = x[i * 512:(i + 1) * 512][None, :].astype(np.float32)
        inp = np.concatenate([ctx, chunk], axis=1)
        p, state = s.run(None, {"input": inp, "state": state, "sr": sr})
        out[i] = p[0, 0]
        ctx = chunk[:, -64:]
    return out


def vad_segments(prob: np.ndarray, thr: float = 0.5, min_speech: float = 0.25, min_silence: float = 0.3, hop: float = 0.032):
    on = prob >= thr
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([i * hop, j * hop]); i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_silence:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [[round(a, 3), round(b, 3)] for a, b in merged if b - a >= min_speech]


def rms_db(x: np.ndarray, hop: int = 160, win: int = 400) -> np.ndarray:
    n = len(x) // hop
    pad = np.pad(x, (win // 2, win))
    idx = np.arange(win)[None, :] + (np.arange(n) * hop)[:, None]
    fr = pad[idx]
    return (20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-5)).astype(np.float32)


def pitch_yin(x: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """f0 (Hz) per 20 ms frame on 8 kHz audio, NaN where unvoiced (energy gate / range edges)."""
    import librosa
    from scipy.signal import resample_poly
    y = resample_poly(x, 1, 2).astype(np.float32)
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=SR // 2, frame_length=512, hop_length=160, trough_threshold=0.1)
    e = rms_db(x, 320)[: len(f0)]
    f0 = f0[: len(e)].astype(np.float32)
    f0[(e < np.percentile(e, 35)) | (f0 >= fmax * 0.98) | (f0 <= fmin * 1.02)] = np.nan
    return f0


def yamnet(x: np.ndarray, model_path: str, groups: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-window (0.975 s) max score per class group -> (times_s, scores[T, G], group names)."""
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components.containers import audio_data as mp_ad
    from mediapipe.tasks.python.core.base_options import BaseOptions
    opts = mp_audio.AudioClassifierOptions(base_options=BaseOptions(model_asset_path=model_path),
                                           running_mode=mp_audio.RunningMode.AUDIO_CLIPS, max_results=-1)
    names = list(groups)
    want = {g: [c.lower() for c in cls] for g, cls in groups.items()}
    times, rows = [], []
    with mp_audio.AudioClassifier.create_from_options(opts) as clf:
        step = SR * 60  # classify in 60 s clips (results carry their own timestamps)
        for off in range(0, len(x), step):
            seg = x[off: off + step].astype(np.float32)
            if len(seg) < int(0.975 * SR):
                break
            ad = mp_ad.AudioData.create_from_array(seg, SR)
            for r in clf.classify(ad):
                cats = {c.category_name.lower(): c.score for c in r.classifications[0].categories}
                times.append(off / SR + r.timestamp_ms / 1000.0)
                rows.append([max([v for k, v in cats.items() if any(w == k or w in k for w in want[g])] or [0.0]) for g in names])
    return np.array(times, np.float32), np.array(rows, np.float32), names


def per_second(values: np.ndarray, hop_s: float, n_sec: int, how: str = "mean") -> np.ndarray:
    out = np.full(n_sec, np.nan, np.float32)
    k = int(round(1 / hop_s))
    for i in range(n_sec):
        v = values[i * k:(i + 1) * k]
        v = v[~np.isnan(v)] if v.dtype.kind == "f" else v
        if len(v):
            out[i] = {"mean": np.mean, "max": np.max, "std": np.std}[how](v)
    return out


def resample_to_seconds(times: np.ndarray, vals: np.ndarray, n_sec: int) -> np.ndarray:
    """Max over windows overlapping each second."""
    out = np.zeros((n_sec, vals.shape[1]), np.float32)
    for t, v in zip(times, vals):
        for s in range(int(t), min(n_sec, int(t + 0.975) + 1)):
            out[s] = np.maximum(out[s], v)
    return out
