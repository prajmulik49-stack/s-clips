"""`joshiclip doctor`: external tools, Python packages, model files, versions."""
from __future__ import annotations

import importlib
import os
import shutil
import sys

from . import tools
from .config import Config

PACKAGES = [  # (import name, pip name, required)
    ("yt_dlp", "yt-dlp", True), ("numpy", "numpy", True), ("scipy", "scipy", True), ("yaml", "PyYAML", True),
    ("faster_whisper", "faster-whisper", True), ("ctranslate2", "ctranslate2", True),
    ("onnxruntime", "onnxruntime", True), ("mediapipe", "mediapipe", True), ("cv2", "opencv-python", True),
    ("scenedetect", "scenedetect", True), ("librosa", "librosa", True), ("soundfile", "soundfile", True),
    ("sklearn", "scikit-learn", True), ("lightgbm", "lightgbm", True), ("rapidfuzz", "RapidFuzz", True),
    ("torch", "torch", True), ("torchaudio", "torchaudio", True), ("PIL", "pillow", True),
    ("demucs", "demucs", False), ("df", "deepfilternet", False), ("pyloudnorm", "pyloudnorm", False),
    ("rapidocr_onnxruntime", "rapidocr-onnxruntime", False),
]

MODEL_FILES = [("silero_vad.onnx", True), ("yamnet.tflite", True), ("blaze_face_short_range.tflite", True),
               ("face_detection_yunet_2023mar.onnx", False)]
MODEL_URLS = {
    "yamnet.tflite": "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
}


def _whisper_cached(name: str) -> str | None:
    try:
        from faster_whisper.utils import download_model
        return download_model(name, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def doctor(cfg: Config) -> int:
    ok = True
    print(f"python   {sys.version.split()[0]}  ({sys.executable})  cpus={os.cpu_count()}")
    print(f"shutil.which('ffmpeg') from this process: {shutil.which('ffmpeg') or 'NOT FOUND'}")
    ff, fp, fc = tools.ffmpeg(cfg["tools"].get("ffmpeg")), tools.ffprobe(), tools.fpcalc(cfg["tools"].get("fpcalc"))
    for t in (ff, fp):
        ok &= bool(t.path)
        print(f"[{'OK ' if t.path else 'ERR'}] {t.name:8s} {t.path or '-'}\n        {t.version}  [{t.source}]")
    if ff.path and ff.missing_filters:
        ok = False
        print(f"[ERR] ffmpeg missing filters: {', '.join(ff.missing_filters)}")
    print(f"[{'OK ' if fc.path else 'opt'}] fpcalc   {fc.path or 'not installed (Chromaprint); built-in landmark fingerprinter is used'}")
    print(f"        yt-dlp gets: {' '.join(tools.ytdlp_args())}")
    print("packages:")
    for mod, pipname, req in PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(f"  [OK ] {pipname:22s} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok &= not req
            print(f"  [{'ERR' if req else 'opt'}] {pipname:22s} missing ({type(e).__name__})  -> pip install {pipname}")
    print("models:")
    mdir = cfg.path("models")
    for f, req in MODEL_FILES:
        p = mdir / f
        have = p.exists()
        ok &= have or not req
        hint = f"  -> download {MODEL_URLS[f]}" if not have and f in MODEL_URLS else ""
        print(f"  [{'OK ' if have else ('ERR' if req else 'opt')}] {f:34s} {p.stat().st_size // 1024 if have else '-'} KB{hint}")
    for name in cfg["asr"]["candidates"]:
        p = _whisper_cached(name)
        chosen = name == cfg["asr"]["model"]
        ok &= bool(p) or not chosen
        print(f"  [{'OK ' if p else ('ERR' if chosen else 'opt')}] faster-whisper {name:19s} {'(selected) ' if chosen else ''}{p or 'not cached (downloads on first use)'}")
    try:
        import torch
        hub = os.path.join(torch.hub.get_dir(), "checkpoints")
        mms = any("mms" in f.lower() for f in os.listdir(hub)) if os.path.isdir(hub) else False
        print(f"  [{'OK ' if mms else 'opt'}] torchaudio MMS_FA aligner         {'cached' if mms else 'downloads on first use (~1.2 GB)'}")
    except Exception:  # noqa: BLE001
        pass
    ref = cfg.path("style_ref")
    print(f"  [{'OK ' if ref.exists() else 'opt'}] style reference                  {ref}")
    print("RESULT:", "OK" if ok else "MISSING REQUIRED ITEMS")
    return 0 if ok else 1
