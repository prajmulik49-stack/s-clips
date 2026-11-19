"""Cluster -> best version; across clusters -> 3-6 Shorts with the mix rule, overlap and region limits.
Also the deterministic title (template + transcript quote, no generated text)."""
from __future__ import annotations

import math
import re

import numpy as np

from .text import hinglish, lex

EMOJI = {"funny": "😂", "laugh": "😂", "danger": "😱", "scream": "😱", "emotional": "🥺", "cry": "😢",
         "reveal": "😳", "gasp": "😳", "question": "🤔", "food": "😋", "cheer": "🥳", "dog": "🐶",
         "conflict": "😤", "name_call": "😳", "energy": "😮", "reveal_cue": "😳"}


def _key(c):
    return (-c["final"], c["start"])


def overlap(a, b) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / min(a["dur"], b["dur"])


def select(A: dict, cands: list[dict], cfg, max_shorts: int | None = None) -> dict:
    s = cfg["selection"]
    alive = [c for c in cands if not c["reject"]]
    clusters: dict[str, list[dict]] = {}
    for c in alive:
        clusters.setdefault(c["cluster"], []).append(c)
    best = []
    cl_info = {}
    for cid in sorted(clusters):
        vs = clusters[cid]
        b = min(vs, key=_key)
        bh = min(vs, key=lambda c: (-c["hook_score"], c["start"]))
        bs = min(vs, key=lambda c: (-c["story_score"], c["start"]))
        cl_info[cid] = {"versions": len(vs), "best": b["id"], "best_hook": bh["id"], "best_story": bs["id"]}
        best.append(b)
    best.sort(key=_key)
    n_target = int(min(max_shorts or s["max_shorts"], max(s["min_shorts"], round(A["duration"] / s["seconds_per_short"]))))
    n_target = min(n_target, max_shorts or s["max_shorts"])
    picked: list[dict] = []
    log = []

    def ok(c):
        if c["final"] < s["min_score"]:
            return "below_min_score"
        for p in picked:
            if overlap(c, p) > s["max_overlap_frac"]:
                return f"overlaps_short_with_cand_{p['id']}"
        reg = int(c["start"] // s["region_s"])
        if sum(1 for p in picked if int(p["start"] // s["region_s"]) == reg) >= s["max_per_region"]:
            return f"region_{reg}_full"
        return None

    # mix rule: overall #1 by hook and #1 by story (if they pass min_score)
    for field, why in (("hook_score", "mix_rule_best_hook"), ("story_score", "mix_rule_best_story")):
        pool = sorted(best, key=lambda c: (-c[field], c["start"]))
        if pool and len(picked) < n_target:
            c = pool[0]
            if c in picked:
                continue
            r = ok(c)
            if r is None:
                c["picked_by"] = why; picked.append(c)
            else:
                log.append({"cand": c["id"], "skip": r, "for": why})
    for c in best:
        if len(picked) >= n_target:
            break
        if c in picked:
            continue
        r = ok(c)
        if r is None:
            c["picked_by"] = "score"; picked.append(c)
        else:
            log.append({"cand": c["id"], "skip": r})
    picked.sort(key=lambda c: c["start"])
    for c in picked:
        d = c["hook_score"] - c["story_score"]
        c["tag"] = "hook" if d > 0.1 else "story" if d < -0.1 else "balanced"
        c["title"] = make_title(A, c)
    return {"target_count": n_target, "clusters": cl_info, "winners": [c["id"] for c in picked], "skips": log}


# ---------------------------------------------------------------- titles
def _title_case(r: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in r.split())


def moment_type(c: dict) -> str:
    types = c.get("hook_types") or []
    for t in ("danger", "funny", "emotional", "reveal", "question", "conflict"):
        if t in types:
            return t
    pt = c.get("peak_type")
    return {"laugh": "funny", "scream": "danger", "cheer": "cheer", "gasp": "reveal", "reveal": "reveal",
            "energy": "reveal"}.get(pt, "funny")


def make_title(A: dict, c: dict, max_chars: int = 40) -> str:
    """Quote the most quotable line near the peak (3-7 words, romanized, Title Case) + 1-2 emojis by moment type."""
    sents = A["sentences"][c["si"]:c["ei"] + 1]
    pt = c["peak_t"] if c["peak_t"] is not None else (c["start"] + c["end"]) / 2
    ps = A["per_sec"]
    best, bkey = None, None
    for s in sents:
        rom = re.sub(r"[^\w\s?!']", "", hinglish.romanize(s["text"])).strip()
        n = len(rom.split())
        if not (3 <= n <= 8) or len(rom) > max_chars - 4:
            continue
        if lex.first_word_dependent(s["text"]) or lex.hits("intro_outro", s["text"]):
            continue
        dist = abs((s["s"] + s["e"]) / 2 - pt)
        energy = float(ps["rms"][int(s["s"]):int(s["e"]) + 1].mean())
        k = (round(dist / 10 - energy / 20 - (0.5 if s["complete"] else 0), 6), s["s"])
        if bkey is None or k < bkey:
            best, bkey = rom, k
    mt = moment_type(c)
    emo = EMOJI.get(mt, "😂")
    second = {"danger": "😢", "funny": "🤣", "reveal": "😮"}.get(mt, "")
    if best is None:
        keys = [w for s in sents for w in hinglish.romanize(s["text"]).split()
                if lex.key_word(w) not in {lex.key_word(x) for x in lex.lexicon().get("stopwords", [])} and len(w) > 2]
        best = " ".join(keys[:4]) or "Ye Dekho"
    t = _title_case(best.rstrip("?!")) + ("?" if best.endswith("?") else "")
    while len(t) + 3 > max_chars and " " in t:
        t = t.rsplit(" ", 1)[0]
    return f"{t} {emo}{second}" if len(t) + 4 <= max_chars else f"{t} {emo}"
