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


def from_youtube_json3(path: Path) -> dict:
    """YouTube auto-captions (json3) -> segments with word timings. Word end = next word start (capped)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = []
    for ev in d.get("events", []):
        parts = [s for s in ev.get("segs", []) or [] if s.get("utf8", "").strip()]
        if not parts:
            continue
        t0 = ev["tStartMs"] / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        ws = [{"w": p["utf8"].strip(), "s": round(t0 + p.get("tOffsetMs", 0) / 1000.0, 3), "p": 0.9} for p in parts]
        for i, w in enumerate(ws):
            nxt = ws[i + 1]["s"] if i + 1 < len(ws) else t0 + dur
            w["e"] = round(max(w["s"] + 0.08, min(nxt, w["s"] + 1.2)), 3)
        segs.append({"s": ws[0]["s"], "e": ws[-1]["e"], "text": " ".join(w["w"] for w in ws),
                     "avg_logprob": 0.0, "no_speech": 0.0, "words": ws})
    # split multi-word tokens ("मैं हूं") into single words with interpolated times
    for s in segs:
        out = []
        for w in s["words"]:
            toks = w["w"].split()
            if len(toks) <= 1:
                out.append(w); continue
            step = (w["e"] - w["s"]) / len(toks)
            for k, t in enumerate(toks):
                out.append({"w": t, "s": round(w["s"] + k * step, 3), "e": round(w["s"] + (k + 1) * step, 3), "p": w["p"]})
        s["words"] = out
    return {"model": "youtube-auto-captions", "model_ref": str(path), "settings": {}, "seconds": 0.0, "rtf": 0.0,
            "segments": segs}


def transcribe_batched(cfg: Config, wav: Path, model_name: str, batch_size: int = 8) -> dict:
    """faster-whisper BatchedInferencePipeline (VAD-chunked, batched) - the fast-mode fallback."""
    from faster_whisper import BatchedInferencePipeline
    audio, sr = sf.read(str(wav), dtype="float32")
    t = time.perf_counter()
    model = load_model(cfg, model_name)
    pipe = BatchedInferencePipeline(model=model)
    a = cfg["asr"]
    segs_it, _ = pipe.transcribe(audio, language=a["language"], beam_size=a["beam_size"], temperature=a["temperature"],
                                 word_timestamps=True, batch_size=batch_size, condition_on_previous_text=False)
    segs = []
    for s in segs_it:
        words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(float(w.probability), 4)}
                 for w in (s.words or []) if w.word.strip()]
        segs.append({"s": round(s.start, 3), "e": round(s.end, 3), "text": s.text.strip(), "avg_logprob": round(float(s.avg_logprob), 4),
                     "no_speech": round(float(s.no_speech_prob), 4), "words": words})
    dt = time.perf_counter() - t
    return {"model": f"{model_name} (batched)", "model_ref": _model_ref(cfg, model_name), "settings": {"batch_size": batch_size},
            "seconds": round(dt, 1), "rtf": round(dt / (len(audio) / sr), 3), "segments": segs}
