"""Virtual candidates (timestamps only) + hard filters with logged reject reasons."""
from __future__ import annotations

import numpy as np

from .text import lex


def _end_quality(s: dict) -> float:
    return (1.0 if s["complete"] else 0.0) + min(1.0, s["pause_after"]) * 0.5 + (0.3 if s["punct_end"] else 0.0)


def generate(A: dict, cfg) -> list[dict]:
    sents, peaks, dur = A["sentences"], A["peaks"], A["duration"]
    c = cfg["candidates"]
    out = []
    for i, s in enumerate(sents):
        start = max(0.0, s["s"] - min(0.15, s["pause_before"] / 2))
        mid = sum(c["len_pref_s"]) / 2
        in_rng = lambda lo, hi: [j for j in range(i, len(sents)) if lo <= sents[j]["e"] - start <= hi]
        pref, allow = in_rng(*c["len_pref_s"]), in_rng(*c["len_allow_s"])
        # clean ending first: complete in 40-50 s, else complete in 35-55 s, else best in 40-50 s
        ends = [j for j in pref if sents[j]["complete"]] or [j for j in allow if sents[j]["complete"]] or pref or allow
        ends.sort(key=lambda j: (-round(_end_quality(sents[j]), 6), abs((sents[j]["e"] - start) - mid), j))
        for j in ends[: c["max_ends_per_start"]]:
            e = sents[j]
            end = min(dur, e["e"] + min(0.25, e["pause_after"] / 2))
            inside = [p for p in peaks if start <= p["t"] < end]
            pk = max(inside, key=lambda p: (p["strength"], -p["t"])) if inside else None
            out.append({"id": len(out), "si": i, "ei": j, "start": round(start, 3), "end": round(end, 3),
                        "dur": round(end - start, 3), "peak": pk["id"] if pk else None,
                        "peak_t": pk["t"] if pk else None, "peak_type": pk["type"] if pk else None,
                        "cluster": f"p{pk['id']:03d}" if pk else f"np{i:04d}"})
    return out


def _words_in(A, a, b):
    return [w for w in A["words"] if a <= w["s"] < b]


def hard_filters(A: dict, cands: list[dict], cfg) -> None:
    """Sets cand['reject'] (first reason) and cand['reject_all'] (every reason). Edge zone handled after scoring."""
    f = cfg["filters"]
    ps, sents, dur = A["per_sec"], A["sentences"], A["duration"]
    ratio = f["phrase_match_ratio"]
    loc_thr = cfg["topics"]["location_hist_threshold"]
    for c in cands:
        rs = []
        a, b = c["start"], c["end"]
        head = " ".join(w["w"] for w in _words_in(A, a, a + f["intro_window_s"]))
        hit = lex.hits("intro_outro", head, ratio)
        if hit:
            rs.append(f"intro_outro_phrase:{hit[0]}")
        body = " ".join(w["w"] for w in _words_in(A, a, b))
        sp = lex.hits("sponsor", body, ratio)
        if sp:
            rs.append(f"sponsor_phrase:{sp[0]}")
        s0 = sents[c["si"]]
        dep = lex.first_word_dependent(s0["text"])
        if dep:
            rs.append(f"dependent_first_word:{dep}")
        if not s0["is_boundary"] and s0["pause_before"] < f["min_pause_before_s"]:
            rs.append("mid_topic_start")
        if not sents[c["ei"]]["complete"]:
            rs.append("incomplete_ending")
        ia, ib = int(a), max(int(a) + 1, int(np.ceil(b)))
        speech = float(ps["speech"][ia:ib].mean())
        noface = float((ps["faces"][ia:ib] < 0.5).mean())
        c["speech_frac"] = round(speech, 4); c["no_face_frac"] = round(noface, 4)
        if speech < f["min_speech_frac"]:
            rs.append(f"low_speech:{speech:.2f}")
        if noface > f["max_no_face_frac"]:
            rs.append(f"no_faces:{noface:.2f}")
        z0, z1 = f["location_change_zone"]
        la, lb = int(a + z0 * (b - a)), int(np.ceil(a + z1 * (b - a)))
        cut = ps["scene_cut"]
        amb = ps["ambience_vec"]
        W = int(cfg["topics"]["location_hist_window_s"])
        def amb_shift(t):
            if t - W < 0 or t + W > len(amb):
                return 0.0
            return float(np.abs(amb[t - W:t].mean(0) - amb[t:t + W].mean(0)).sum())
        locs = [t for t in range(la, min(lb, len(cut))) if ps["location"][t] >= f["location_change_score"]
                and cut[max(0, t - 2):t + 3].max() > 0 and amb_shift(t) >= f["location_ambience_shift"]]
        if locs:
            rs.append(f"location_change@{locs[0]}s")
        c["edge"] = bool(a < f["edge_zone_s"] or b > dur - f["edge_zone_s"])
        c["reject_all"] = rs
        c["reject"] = rs[0] if rs else None


def apply_edge_rule(cands: list[dict], cfg) -> None:
    """Edge-zone candidates survive only if in the top N% of all scored candidates."""
    pct = cfg["filters"]["edge_keep_top_pct"]
    scores = sorted((c["final"] for c in cands), reverse=True)
    if not scores:
        return
    k = max(1, int(np.ceil(len(scores) * pct / 100)))
    cutoff = scores[k - 1]
    for c in cands:
        if c["edge"] and c["final"] < cutoff:
            c["reject_all"].append("edge_zone_not_top5pct")
            c["reject"] = c["reject"] or "edge_zone_not_top5pct"
