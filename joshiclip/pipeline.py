"""Orchestration: ingest -> analyze (cached) -> candidates -> filters -> scoring -> selection -> render -> checks."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

from . import __version__
from .cache import StageCache
from .config import Config, load_lexicon, stable_hash
from .text import lex


def workdir_for(cfg: Config, vid: str) -> Path:
    return cfg.path("work") / vid


def clip_filename(rank: int, title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:64]
    return f"{rank:02d}_{safe or 'moment'}.mp4"


def _dedupe_loops(words: list[dict]) -> list[dict]:
    """Drop Whisper repetition loops: an n-gram (n=1..4) repeated 3+ times back to back keeps one copy."""
    toks = [w["w"] for w in words]
    keep = [True] * len(words)
    i = 0
    while i < len(toks):
        done = False
        for n in (4, 3, 2, 1):
            g = toks[i:i + n]
            if len(g) < n:
                continue
            r = 1
            while toks[i + r * n:i + (r + 1) * n] == g:
                r += 1
            if r >= 3:
                for j in range(i + n, i + r * n):
                    keep[j] = False
                i += r * n; done = True
                break
        if not done:
            i += 1
    return [w for w, k in zip(words, keep) if k]


def analyze(cfg: Config, target: str | None, latest: bool = False, vid: str | None = None, log=print, fast: bool = False) -> dict:
    from . import ingest
    from .analysis import audio_feats as A
    from .analysis import speech, topics, transcript, vision_feats as V
    import soundfile as sf

    lx = load_lexicon(cfg)
    lex.set_lexicon(lx)
    if fast:
        from . import fast as fastmod
        vid, wd, meta = fastmod.ingest_lite(cfg, target, latest, vid, log)
    else:
        v0, src = ingest.resolve_source(cfg, target, latest, log)
        vid = vid or v0
        wd = workdir_for(cfg, vid)
    log(f"[{vid}] analyze  (work dir {wd})")
    t_all = time.perf_counter()
    if not fast:
        meta = ingest.normalize(cfg, src, wd, log)
    C = StageCache(wd)
    C.timings["normalize"] = sum(meta["timings_s"].values())
    nk = stable_hash(meta["key"])
    mdir = cfg.path("models")
    dur = meta["duration"]
    n_sec = int(math.ceil(dur))
    x = None

    def audio():
        nonlocal x
        if x is None:
            x, sr = sf.read(str(wd / "audio_16k_mono.wav"), dtype="float32")
            assert sr == 16000
        return x

    vad_d, vad_a = C.run("vad", 1, {"norm": nk}, lambda: ({"hop": 0.032},
                         {"prob": A.silero_vad(audio(), str(mdir / "silero_vad.onnx"))}), log)
    vad_prob = vad_a["prob"]

    def _transcribe():
        if fast:
            caps = sorted(wd.glob("*.hi-orig.json3")) + sorted(wd.glob("*.hi.json3"))
            if caps:
                log(f"    using YouTube auto-captions {caps[0].name}")
                return transcript.from_youtube_json3(caps[0]), None
            log("    no captions -> batched Whisper " + cfg["fast"]["asr_model"])
            return transcript.transcribe_batched(cfg, wd / "audio_16k_mono.wav", cfg["fast"]["asr_model"]), None
        return transcript.transcribe(cfg, wd / "audio_16k_mono.wav"), None
    tr, _ = C.run("transcript", 2 if fast else 1, {"norm": nk, "asr": cfg["asr"], **({"fast": {k: cfg["fast"][k] for k in ("asr_model", "sub_langs")}} if fast else {})}, _transcribe, log)

    def _align():
        segs = tr["segments"]
        stats = {"engine": "none"}
        if cfg["align"]["engine"] == "mms_fa" and not fast:
            try:
                segs, stats = speech.mms_align(audio(), segs, cfg["align"]["max_shift_s"], log)
            except Exception as e:  # noqa: BLE001
                stats = {"engine": "none", "error": f"{type(e).__name__}: {e}"}
                log(f"    forced alignment unavailable ({stats['error']}); keeping Whisper word times")
        words = [dict(w, seg=k, seg_last=(j == len(s["words"]) - 1)) for k, s in enumerate(segs) for j, w in enumerate(s["words"])]
        words.sort(key=lambda w: (w["s"], w["e"]))
        n0 = len(words)
        words = _dedupe_loops(words)
        # drop words Whisper placed in clear non-speech with low confidence (hallucinations)
        hop = 0.032
        def vmean(w):
            a, b = int(w["s"] / hop), max(int(w["s"] / hop) + 1, int(w["e"] / hop))
            return float(vad_prob[a:b].mean()) if a < len(vad_prob) else 0.0
        words = [w for w in words if not (vmean(w) < 0.15 and w["p"] < 0.5)]
        stats["dropped_loops_or_nonspeech"] = n0 - len(words)
        stats["vad_snapped_edges"] = speech.vad_snap(words, vad_prob, cfg["align"]["vad_threshold"])
        for i, w in enumerate(words):
            w["i"] = i
        return {"stats": stats, "words": words}, None
    al, _ = C.run("align", 2, {"tr": C.keys["transcript"], "align": cfg["align"], "vad": C.keys["vad"], **({"fast": True} if fast else {})}, _align, log)
    words = al["words"]

    def _audio_feats():
        xx = audio()
        rms = A.rms_db(xx, 160)
        f0 = A.pitch_yin(xx, cfg["features"]["pitch_fmin"], cfg["features"]["pitch_fmax"])
        return {"rms_hop": 0.01, "f0_hop": 0.02}, {"rms": rms, "f0": f0}
    _, af = C.run("audio_feats", 1, {"norm": nk, "f": {k: cfg["features"][k] for k in ("pitch_fmin", "pitch_fmax")}},
                  _audio_feats, log)

    def _yam():
        t, s, names = A.yamnet(audio(), str(mdir / "yamnet.tflite"), cfg["features"]["yamnet_classes"])
        return {"names": names}, {"t": t, "scores": s}
    yd, ya = C.run("yamnet", 1, {"norm": nk, "cls": cfg["features"]["yamnet_classes"]}, _yam, log)

    vf = cfg["features"]
    fmodel = {"yunet": "face_detection_yunet_2023mar.onnx", "mediapipe": "blaze_face_short_range.tflite"}[vf["face_detector"]]
    _, vis = C.run("vision", 2, {"norm": nk, "fps": vf["vision_fps"], "conf": vf["face_min_confidence"], "det": vf["face_detector"]},
                   lambda: ({}, V.analyze(wd / "proxy_360p.mp4", str(mdir / fmodel), vf["vision_fps"],
                                          vf["face_min_confidence"], vf["face_detector"])), log)
    sc, _ = C.run("scenes", 1, {"norm": nk, "thr": vf["scene_threshold"]},
                  lambda: ({"cuts": V.scene_cuts(wd / "proxy_360p.mp4", vf["scene_threshold"])}, None), log)

    lex_h = stable_hash(lx)
    se, _ = C.run("sentences", 1, {"al": C.keys["align"], "s": cfg["sentences"], "lex": lex_h},
                  lambda: ({"sentences": speech.build_sentences(words, cfg["sentences"])}, None), log)
    sents = se["sentences"]

    def _diar():
        t = time.perf_counter()
        labels, st = speech.diarize(audio(), sents, af["f0"], 0.02, cfg["diarization"])
        st["seconds"] = round(time.perf_counter() - t, 2)
        return {"labels": labels, "stats": st}, None
    di, _ = C.run("diarization", 1, {"se": C.keys["sentences"], "af": C.keys["audio_feats"], "d": cfg["diarization"]}, _diar, log)
    for s, l in zip(sents, di["labels"]):
        s["spk"] = int(l)

    # ---- per-second feature table
    def _per_sec():
        rms = af["rms"]; f0 = af["f0"]
        ps = {"rms": A.per_second(rms, 0.01, n_sec, "mean"), "rms_max": A.per_second(rms, 0.01, n_sec, "max")}
        semi = 12 * np.log2(np.where(np.isnan(f0), np.nan, f0) / 100.0)
        ps["pitch_std"] = np.nan_to_num(A.per_second(semi, 0.02, n_sec, "std"), nan=0.0)
        ps["pitch_med"] = A.per_second(semi, 0.02, n_sec, "mean")
        ps["speech"] = np.nan_to_num(A.per_second((vad_prob >= cfg["align"]["vad_threshold"]).astype(np.float32), 0.032, n_sec), nan=0.0)
        wps = np.zeros(n_sec, np.float32)
        for w in words:
            if int(w["s"]) < n_sec:
                wps[int(w["s"])] += 1
        ps["words"] = wps
        ys = A.resample_to_seconds(ya["t"], ya["scores"], n_sec)
        for j, nme in enumerate(yd["names"]):
            ps[nme] = ys[:, j]
        vfps = float(vis["fps"])
        fc = vis["face_count"].astype(np.float32)
        ps["faces"] = np.array([fc[int(i * vfps):int((i + 1) * vfps)].mean() if int(i * vfps) < len(fc) else 0 for i in range(n_sec)], np.float32)
        mo = vis["motion"]
        ps["motion"] = np.array([mo[int(i * vfps):int((i + 1) * vfps)].mean() if int(i * vfps) < len(mo) else 0 for i in range(n_sec)], np.float32)
        ps["location"] = topics.location_curve(vis["hist"], vfps, n_sec, cfg["topics"]["location_hist_window_s"])
        cut = np.zeros(n_sec, np.float32)
        for c in sc["cuts"]:
            if int(c) < n_sec:
                cut[int(c)] = 1
        ps["scene_cut"] = cut
        amb = np.stack([ys[:, j] for j in range(ys.shape[1])] + [np.clip((ps["rms"] + 60) / 60, 0, 1)], 1)
        ps["ambience_vec"] = amb.astype(np.float32)
        return {"n_sec": n_sec}, ps
    _, ps = C.run("per_second", 1, {k: C.keys[k] for k in ("vad", "align", "audio_feats", "yamnet", "vision", "scenes")}
                  | {"thr": cfg["align"]["vad_threshold"], "lw": cfg["topics"]["location_hist_window_s"]}, _per_sec, log)

    tp, _ = C.run("topics", 2, {"se": C.keys["sentences"], "ps": C.keys["per_second"], "t": cfg["topics"], "lex": lex_h},
                  lambda: (topics.boundaries(sents, words, ps, ps["location"], cfg["topics"]), None), log)
    for i, s in enumerate(sents):
        s["boundary_score"] = tp["boundary_score"][i]; s["is_boundary"] = tp["is_boundary"][i]
        s["topic"] = tp["topic_id"][i]; s["location_score"] = tp["location_score"][i]

    comments = None
    cpath = wd / "comments_per_sec.npy"
    if cfg.get("viewers", {}).get("enabled", False):
        from . import viewers
        url = meta.get("source") if fast else (f"https://www.youtube.com/watch?v={vid}" if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None)
        viewers.build(cfg, url, wd, dur, log)
    if cpath.exists():
        comments = np.load(cpath)
    pk, _ = C.run("peaks", 1, {"ps": C.keys["per_second"], "al": C.keys["align"], "p": cfg["peaks"], "lex": lex_h,
                               "comments": stable_hash(comments.tolist()) if comments is not None else None},
                  lambda: ({"peaks": topics.peaks(ps, words, comments, cfg["peaks"])}, None), log)

    C.timings["analyze_total_wall"] = round(time.perf_counter() - t_all, 1)
    (wd / "timings.json").write_text(json.dumps(C.timings, indent=1), encoding="utf-8")
    log(f"  {len(words)} words, {len(sents)} sentences, {di['stats'].get('speakers')} speakers, "
        f"{sum(tp['is_boundary'])} topic boundaries, {len(sc['cuts'])} scene cuts, {len(pk['peaks'])} peaks")
    return {"vid": vid, "workdir": wd, "meta": meta, "fps": meta["fps"], "duration": dur, "n_sec": n_sec,
            "words": words, "sentences": sents, "per_sec": ps, "peaks": pk["peaks"], "topics": tp,
            "scene_cuts": sc["cuts"], "vision": vis, "transcript_model": tr["model"], "align_stats": al["stats"],
            "diar_stats": di["stats"], "cache": C, "lexicon": lx, "version": __version__, "vad_prob": vad_prob,
            "comments_per_sec": comments, "rms10": af["rms"]}


def plan(cfg: Config, A: dict, max_shorts: int | None = None):
    from . import candidates, scoring, selection
    # Real ingests opt into the moment-first planner. Synthetic/minimal analysis
    # fixtures and old cached plans retain the legacy sentence-window contract.
    if cfg.get("engine", {}).get("mode") == "moment_first" and A.get("workdir") is not None:
        from . import moment_first
        return moment_first.plan(A, cfg, max_shorts)
    cands = candidates.generate(A, cfg)
    candidates.hard_filters(A, cands, cfg)
    scoring.score_all(A, cands, cfg)
    candidates.apply_edge_rule(cands, cfg)
    sel = selection.select(A, cands, cfg, max_shorts)
    return cands, sel


def _debug_labels(c: dict, span) -> str:
    """--preview-debug only: scores, hook type, peak marker burned in. Never used for final renders."""
    font = "C\:/Windows/Fonts/arial.ttf"
    txt = (f"cand {c['id']}  final {c['final']:.3f}  hook {c['hook_score']:.3f}  story {c['story_score']:.3f}  "
           f"[{c['tag']}] {','.join(c['hook_types']) or '-'}").replace(":", "\:")
    dt = f"drawtext=fontfile='{font}':text='{txt}':x=20:y=20:fontsize=h/28:fontcolor=yellow:box=1:boxcolor=black@0.6"
    if c["peak_t"] is not None:
        pt = c["peak_t"] - span.start
        dt += (f",drawtext=fontfile='{font}':text='PEAK {c['peak_type']}':x=20:y=h-th-30:fontsize=h/18:fontcolor=red:"
               f"box=1:boxcolor=black@0.6:enable='between(t,{pt - 1:.2f},{pt + 1:.2f})'")
    return dt


def run(cfg: Config, target, latest=False, vid=None, mode="draft", max_shorts=None, export_candidates=False,
        preview_debug=False, render=True, plain=False, log=print) -> int:
    from . import cut, media, report, sync
    A = analyze(cfg, target, latest, vid, log)
    C = A["cache"]
    t = time.perf_counter()
    cands, sel = plan(cfg, A, max_shorts)
    C.timings["plan"] = round(time.perf_counter() - t, 2)
    out_dir = cfg.path("out") / A["vid"]
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = A["workdir"] / "norm.mp4"
    fps = A["fps"]
    preset = cfg["render"][mode]
    h = min(preset["height"], A["meta"]["height"])
    renders = {}
    if render:
        for k, cid in enumerate(sel["winners"], 1):
            c = cands[cid]
            span = cut.snap(c["start"], c["end"], fps)
            c["file"] = f"debug_short_{k:02d}.mp4" if preview_debug else clip_filename(k, c["title"])
            out = out_dir / c["file"]
            t = time.perf_counter()
            vrep = None
            if cfg["layout"]["vertical"] and not preview_debug and not plain:
                from . import vertical
                vrep = vertical.render(A, c, out, cfg, mode, log)
            else:
                vf = _debug_labels(c, span) if preview_debug else ""
                media.run(cut.cut_plain(norm, span, out, preset, h, cfg["render"]["x264_threads"], vf))
            C.timings[f"render_short_{k:02d}"] = round(time.perf_counter() - t, 1)
            t = time.perf_counter()
            vg = vrep.pop("_vgeom", None) if vrep else None
            ch = sync.check_render(out, norm, (span.start, span.end), fps, cfg["sync"]["probe_len_s"], cfg["sync"]["max_av_offset_ms"], vg)
            C.timings[f"check_short_{k:02d}"] = round(time.perf_counter() - t, 1)
            renders[cid] = {"checks": ch, "vertical": vrep}
            log(f"  short_{k:02d} {span.start:.2f}-{span.end:.2f}s  {c['title']}  sync {'OK' if ch['ok'] else 'FAIL'} "
                f"(A/V {ch['max_abs_av_offset_ms']} ms, frames {ch['frames']}/{ch['expected_frames']})")
    if export_candidates:
        ed = out_dir / "candidates"
        ed.mkdir(exist_ok=True)
        ep = cfg["render"]["export_candidates"]
        for c in cands:
            if c["reject"]:
                continue
            span = cut.snap(c["start"], c["end"], fps)
            media.run(cut.cut_plain(norm, span, ed / f"cand_{c['id']:04d}_{c['final']:.3f}.mp4", ep, ep["height"],
                                    cfg["render"]["x264_threads"]))
    rep = report.build(A, cands, sel, cfg, renders)
    report.write(rep, out_dir, C.timings)
    log(f"  report: {out_dir / 'report.md'}")
    from . import upload
    from .viewers import vlog_title
    log(f"  upload pack: {upload.write(rep, cfg, out_dir, A['meta'].get('title') or vlog_title(A['workdir']))}")
    bad = [r for r in renders.values() if not r["checks"]["ok"]]
    return 1 if bad else 0
