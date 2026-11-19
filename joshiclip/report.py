"""report.json (deterministic: no timings, no absolute paths) + report.md + `explain`."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def build(A: dict, cands: list[dict], sel: dict, cfg, renders: dict) -> dict:
    import faster_whisper, lightgbm, mediapipe, onnxruntime, scenedetect, torch, torchaudio  # noqa: E401
    win = {cid: True for cid in sel["winners"]}
    shorts = []
    for k, cid in enumerate(sel["winners"], 1):
        c = cands[cid]
        first5 = " ".join(w["w"] for w in A["words"] if c["start"] <= w["s"] < c["start"] + 5)
        from .text import hinglish
        shorts.append({"n": k, "file": f"short_{k:02d}.mp4", "cand": cid, "start": c["start"], "end": c["end"],
                       "dur": c["dur"], "tag": c["tag"], "title": c["title"], "picked_by": c["picked_by"],
                       "final": c["final"], "hook_score": c["hook_score"], "story_score": c["story_score"],
                       "hook_types": c["hook_types"], "reasons": c["reasons"], "cluster": c["cluster"],
                       "peak_t": c["peak_t"], "peak_type": c["peak_type"],
                       "first5s": first5, "first5s_roman": hinglish.romanize(first5),
                       "checks": renders.get(cid, {}).get("checks")})
    rej = {}
    for c in cands:
        if c["reject"]:
            r = c["reject"].split(":")[0].split("@")[0]
            rej[r] = rej.get(r, 0) + 1
    return {
        "video_id": A["vid"], "duration_s": round(A["duration"], 3), "fps": A["fps"], "engine_version": A["version"],
        "models": {"asr": A["transcript_model"], "asr_settings": {k: cfg["asr"][k] for k in ("compute_type", "beam_size", "temperature", "language")},
                   "aligner": A["align_stats"].get("engine"), "vad": "silero_vad.onnx (v5)", "audio_events": "YAMNet (MediaPipe tflite)",
                   "faces": "MediaPipe BlazeFace short-range", "scenes": "PySceneDetect ContentDetector",
                   "versions": {"faster_whisper": faster_whisper.__version__, "mediapipe": mediapipe.__version__,
                                "onnxruntime": onnxruntime.__version__, "scenedetect": scenedetect.__version__,
                                "torch": torch.__version__, "torchaudio": torchaudio.__version__, "lightgbm": lightgbm.__version__}},
        "lexicon_source": A["lexicon"].get("source"), "scoring": "hand-weighted fallback (ranker not trained yet)",
        "analysis": {"words": len(A["words"]), "sentences": len(A["sentences"]), "speakers": A["diar_stats"].get("speakers"),
                     "topic_boundaries": sum(A["topics"]["is_boundary"]), "scene_cuts": len(A["scene_cuts"]),
                     "peaks": A["peaks"], "align": A["align_stats"]},
        "counts": {"candidates": len(cands), "survivors": sum(1 for c in cands if not c["reject"]),
                   "clusters": len(sel["clusters"]), "target_shorts": sel["target_count"], "rejects_by_reason": dict(sorted(rej.items()))},
        "shorts": shorts, "selection_skips": sel["skips"], "clusters": sel["clusters"],
        "candidates": [{k: c.get(k) for k in ("id", "start", "end", "dur", "cluster", "peak", "peak_t", "peak_type", "edge",
                                             "speech_frac", "no_face_frac", "hook_score", "story_score", "viewer_score",
                                             "final", "hook_types", "reject", "reject_all", "hook_features", "story_features",
                                             "reasons")} | {"winner": c["id"] in win} for c in cands],
    }


def write(rep: dict, out_dir: Path, timings: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1, sort_keys=True), encoding="utf-8")
    L = [f"# JoshiClip report — {rep['video_id']}", "",
         f"Video {_fmt_t(rep['duration_s'])} at {rep['fps']} fps · ASR {rep['models']['asr']} · aligner {rep['models']['aligner']} · "
         f"lexicon `{rep['lexicon_source']}` · scoring: {rep['scoring']}", "",
         f"**{rep['counts']['candidates']} candidates → {rep['counts']['survivors']} survive filters → "
         f"{rep['counts']['clusters']} moment clusters → {len(rep['shorts'])} Shorts** (target {rep['counts']['target_shorts']})", "",
         "## Shorts", "", "| # | span | dur | type | final | hook | story | title | picked by |", "|---|---|---|---|---|---|---|---|---|"]
    for s in rep["shorts"]:
        L.append(f"| {s['n']} | {_fmt_t(s['start'])}–{_fmt_t(s['end'])} | {s['dur']:.1f}s | {s['tag']} | {s['final']:.3f} | "
                 f"{s['hook_score']:.3f} | {s['story_score']:.3f} | {s['title']} | {s['picked_by']} |")
    for s in rep["shorts"]:
        L += ["", f"### short_{s['n']:02d} — {s['title']}", "",
              f"- span {_fmt_t(s['start'])}–{_fmt_t(s['end'])} ({s['dur']:.1f}s), cluster `{s['cluster']}`, peak "
              f"{_fmt_t(s['peak_t']) if s['peak_t'] is not None else '-'} ({s['peak_type']}), hook types: {', '.join(s['hook_types']) or '-'}",
              f"- top reasons: " + "; ".join(s["reasons"]),
              f"- first 5 s: {s['first5s_roman']}", f"- first 5 s (Devanagari): {s['first5s']}"]
        ch = s.get("checks")
        if ch:
            L.append(f"- checks: fps {ch['out_fps']:.0f}/{ch['expected_fps']} · frames {ch['frames']}/{ch['expected_frames']} · "
                     f"max A/V offset {ch['max_abs_av_offset_ms']} ms ({ch['n_reliable_probes']} probes) · "
                     f"max audio lag {ch['max_abs_audio_lag_ms']} ms · {'PASS' if ch['ok'] else 'FAIL'}")
    L += ["", "## Rejections", "", "| reason | candidates |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep["counts"]["rejects_by_reason"].items()]
    L += ["", "## Peaks", "", "| t | type | strength |", "|---|---|---|"]
    L += [f"| {_fmt_t(p['t'])} | {p['type']} | {p['strength']:.2f} |" for p in rep["analysis"]["peaks"]]
    L += ["", "## All candidates (top 40 by final score)", "", "| id | span | final | hook | story | cluster | status |", "|---|---|---|---|---|---|---|"]
    for c in sorted(rep["candidates"], key=lambda c: (-c["final"], c["start"]))[:40]:
        st = "**WINNER**" if c["winner"] else (c["reject"] or "survived")
        L.append(f"| {c['id']} | {_fmt_t(c['start'])}–{_fmt_t(c['end'])} | {c['final']:.3f} | {c['hook_score']:.3f} | "
                 f"{c['story_score']:.3f} | {c['cluster']} | {st} |")
    L += ["", "## Stage timings (CPU, seconds)", "", "| stage | s |", "|---|---|"]
    L += [f"| {k} | {v:.1f} |" for k, v in timings.items()]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def explain(path: str) -> int:
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"{rep['video_id']}: {rep['counts']}")
    for s in rep["shorts"]:
        print(f"\nshort_{s['n']:02d} [{s['tag']}] {s['start']:.2f}-{s['end']:.2f}s  final={s['final']:.3f} "
              f"hook={s['hook_score']:.3f} story={s['story_score']:.3f}\n  title: {s['title']}\n  why: " + "; ".join(s["reasons"]))
        c = rep["candidates"][s["cand"]]
        print("  hook features:  " + ", ".join(f"{k}={v:.2f}" for k, v in c["hook_features"].items()))
        print("  story features: " + ", ".join(f"{k}={v:.2f}" for k, v in c["story_features"].items()))
    return 0
