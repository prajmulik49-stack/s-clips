"""Speech-to-text (faster-whisper, word timestamps). Whisper is used ONLY for transcription."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import ROOT, Config


def _model_ref(cfg: Config, name: str) -> str:
    p = (cfg["asr"].get("model_paths") or {}).get(name)
    if p:
        pp = Path(p) if Path(p).is_absolute() else ROOT / p
        if pp.exists():
            return str(pp)
    return name


def load_model(cfg: Config, name: str | None = None):
    from faster_whisper import WhisperModel
    a = cfg["asr"]
    return WhisperModel(_model_ref(cfg, name or a["model"]), device="cpu", compute_type=a["compute_type"],
                        cpu_threads=a["cpu_threads"], num_workers=1)


def _transcribe_array(model, audio: np.ndarray, a: dict, offset: float = 0.0) -> list[dict]:
    segs, _info = model.transcribe(
        audio, language=a["language"], task="transcribe", beam_size=a["beam_size"], temperature=a["temperature"],
        word_timestamps=True, vad_filter=a["vad_filter"], condition_on_previous_text=a["condition_on_previous_text"])
    out = []
    for s in segs:
        words = [{"w": w.word.strip(), "s": round(w.start + offset, 3), "e": round(w.end + offset, 3),
                  "p": round(float(w.probability), 4)} for w in (s.words or []) if w.word.strip()]
        out.append({"s": round(s.start + offset, 3), "e": round(s.end + offset, 3), "text": s.text.strip(),
                    "avg_logprob": round(float(s.avg_logprob), 4), "no_speech": round(float(s.no_speech_prob), 4),
                    "words": words})
    return out


def transcribe(cfg: Config, wav: Path) -> dict:
    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == 16000
    t = time.perf_counter()
    model = load_model(cfg)
    segs = _transcribe_array(model, audio, cfg["asr"])
    dt = time.perf_counter() - t
    return {"model": cfg["asr"]["model"], "model_ref": _model_ref(cfg, cfg["asr"]["model"]),
            "settings": {k: cfg["asr"][k] for k in ("compute_type", "cpu_threads", "language", "beam_size", "temperature",
                                                     "vad_filter", "condition_on_previous_text")},
            "seconds": round(dt, 1), "rtf": round(dt / (len(audio) / sr), 3), "segments": segs}


def bench(cfg: Config, workdir: Path) -> None:
    """RTF + agreement on an excerpt for each candidate model -> research/ASR_BENCHMARK.md."""
    audio, sr = sf.read(str(workdir / "audio_16k_mono.wav"), dtype="float32")
    t0, t1 = cfg["asr"]["bench_excerpt_s"]
    x = audio[int(t0 * sr):int(t1 * sr)]
    rows, texts = [], {}
    for name in cfg["asr"]["candidates"]:
        t = time.perf_counter()
        m = load_model(cfg, name)
        tl = time.perf_counter() - t
        t = time.perf_counter()
        segs = _transcribe_array(m, x, cfg["asr"], t0)
        dt = time.perf_counter() - t
        words = [w for s in segs for w in s["words"]]
        texts[name] = " ".join(w["w"] for w in words)
        rows.append({"model": name, "load_s": round(tl, 1), "seconds": round(dt, 1), "rtf": round(dt / (t1 - t0), 3),
                     "words": len(words), "mean_word_prob": round(float(np.mean([w["p"] for w in words])) if words else 0, 3)})
        del m
        print(rows[-1])
    from rapidfuzz.distance import Levenshtein
    names = list(texts)
    agree = 1 - Levenshtein.normalized_distance(texts[names[0]].split(), texts[names[-1]].split()) if len(names) > 1 else 1.0
    out = cfg.path("research") / "ASR_BENCHMARK.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# ASR benchmark ({workdir.name}, excerpt {t0}-{t1} s, CPU int8, {cfg['asr']['cpu_threads']} threads)", "",
             "| model | load s | transcribe s | RTF | words | mean word prob |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['load_s']} | {r['seconds']} | {r['rtf']} | {r['words']} | {r['mean_word_prob']} |" for r in rows]
    lines += ["", f"Word-level agreement {names[0]} vs {names[-1]}: {agree:.3f} (1 - normalized word edit distance)", ""]
    for n in names:
        lines += [f"**{n}:** {texts[n][:600]}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    (workdir / "asr_bench.json").write_text(json.dumps({"rows": rows, "agreement": agree}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
