"""Moment-first event map, incident cards, titleability gate, and selection."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .text import hinglish, lex

CATEGORY_KEYS = {
    "drama_relationship": ("saas", "bahu", "taunt", "gussa", "daant", "ladai", "jhagda", "rishta"),
    "festival_emotional": ("birthday", "diwali", "holi", "rakhi", "puja", "emotional", "rona", "yaad", "miss"),
    "dance": ("dance", "naach", "nach", "thumka", "gaana", "song"),
    "funny_accident": ("funny", "hasi", "hahaha", "gir", "gira", "palat", "bheeg", "splash", "accident"),
    "money_numbers": ("rupee", "rs", "lakh", "crore", "paisa", "price", "bill"),
    "danger_loss_fear": ("bach gaye", "danger", "darr", "dar", "kho", "gaya", "gayi", "chot", "takra"),
    "travel_scenery": ("trip", "travel", "vacation", "pahad", "mountain", "beach", "flight", "hotel"),
}
EMOJI = {"funny_accident": "😂", "danger_loss_fear": "😱", "festival_emotional": "🥺",
         "drama_relationship": "😤", "dance": "💃", "money_numbers": "💰", "travel_scenery": "🌄"}


def _text(A, a, b):
    return " ".join(w["w"] for w in A["words"] if a <= w["s"] < b)


def _sentence_index(sents, t):
    for i, s in enumerate(sents):
        if s["s"] <= t < s["e"]:
            return i
    return max(0, min(len(sents) - 1, len([s for s in sents if s["s"] <= t]) - 1))


def event_map(A):
    """Split by transcript boundaries, scene cuts, and ambience changes."""
    sents, cuts = A.get("sentences", []), set(int(x) for x in A.get("scene_cuts", []))
    if not sents:
        return []
    events, start = [], 0
    for i in range(1, len(sents)):
        prior, cur = sents[i - 1], sents[i]
        boundary = cur.get("is_boundary", False) or any(prior["e"] <= c <= cur["s"] for c in cuts)
        if boundary and cur["s"] - sents[start]["s"] >= 6:
            events.append(_event(sents, start, i - 1))
            start = i
    events.append(_event(sents, start, len(sents) - 1))
    return events


def _event(sents, a, b):
    return {"id": a, "si": a, "ei": b, "start": sents[a]["s"], "end": sents[b]["e"],
            "topic": sents[a].get("topic"), "text": " ".join(x["text"] for x in sents[a:b + 1])}


def _category(text, peak_type):
    low = hinglish.romanize(text).lower()
    scores = {cat: sum(1 for k in keys if lex.phrase_in(k, low, 75)) for cat, keys in CATEGORY_KEYS.items()}
    if peak_type in ("scream", "gasp"):
        scores["danger_loss_fear"] += 1
    if peak_type in ("laugh",):
        scores["funny_accident"] += 1
    return max(scores, key=lambda x: (scores[x], x)) if max(scores.values(), default=0) else "funny_accident"


def _near_names(text):
    names = lex.lexicon().get("names") or []
    return [str(n) for n in names if lex.phrase_in(str(n), text, 100)]


def _what(text, category):
    low = hinglish.romanize(text).lower()
    for phrase in CATEGORY_KEYS.get(category, ()):
        if lex.phrase_in(phrase, low, 75):
            return phrase
    words = [hinglish.romanize(x) for x in re.findall(r"[\\w]+", text)]
    stop = set(lex.lexicon().get("stopwords") or [])
    for word in words:
        if len(word) >= 4 and word.lower() not in stop:
            return word.lower()
    return None


def _comment_score(A, a, b):
    comments = A.get("comments_per_sec")
    if comments is None or np.sum(comments) <= 0:
        return 0.0
    x = np.asarray(comments)
    local = float(x[int(a):max(int(a) + 1, int(np.ceil(b)))].sum() / max(1, b - a))
    return float(np.clip(local / (np.percentile(x, 95) + 1e-9), 0, 1))


def _consensus_prior(cfg, category):
    path = cfg.path("research") / cfg["research"].get("report_file", "report.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        vals = []
        for role in report.get("categories", {}).values():
            if category in role:
                vals.append(role[category].get("median_normalized_views", 0))
        return float(np.clip(np.mean(vals), 0, 1)) if vals else 0.0
    except (OSError, json.JSONDecodeError, TypeError):
        return 0.0


def incident_cards(A, cfg):
    """Create titleable cards from audio peaks and incident words inside events."""
    ps, sents = A["per_sec"], A["sentences"]
    cards = []
    events = event_map(A)
    for event in events:
        peaks = [p for p in A.get("peaks", []) if event["start"] <= p["t"] <= event["end"]]
        if not peaks:
            # A loud/keyword event without a YAMNet peak still gets one incident anchor.
            peaks = [{"id": None, "t": (event["start"] + event["end"]) / 2, "type": "energy", "strength": 0.5}]
        for peak in sorted(peaks, key=lambda p: (-p.get("strength", 0), p["t"]))[:3]:
            t = float(peak["t"])
            nearby = _text(A, max(event["start"], t - 8), min(event["end"], t + 8))
            category = _category(nearby, peak.get("type"))
            who = _near_names(nearby)
            what = _what(nearby, category)
            shout = peak.get("type") in ("scream", "laugh", "cheer", "gasp") and peak.get("strength", 0) >= cfg["engine"].get("must_peak_strength", 3.0)
            if (not who or not what) and not shout:
                continue
            what = what or peak.get("type")
            center = _sentence_index(sents, t)
            start_i = center
            while start_i > event["si"] and sents[start_i]["s"] - sents[start_i - 1]["s"] < 2.0 and sents[start_i]["s"] - sents[center]["s"] > -12:
                start_i -= 1
            end_i = center
            while end_i < event["ei"] and sents[end_i]["e"] < t + 4:
                end_i += 1
            a, b = max(event["start"], sents[start_i]["s"] - 0.15), min(event["end"], sents[end_i]["e"] + 0.25)
            build = b - a
            if build >= cfg["engine"]["story_min_s"]:
                fmt, lo, hi = "story", cfg["engine"]["story_min_s"], cfg["engine"]["story_max_s"]
                b = min(event["end"], a + hi)
            else:
                fmt, lo, hi = "incident", cfg["engine"]["incident_min_s"], cfg["engine"]["incident_max_s"]
                a = max(event["start"], t - hi * 0.65)
                b = min(event["end"], max(t + 3, a + lo))
            duration = b - a
            if duration < lo or duration > hi + 0.5:
                continue
            ia, ib = int(a), min(len(ps["rms"]), int(np.ceil(b)))
            rms = np.asarray(ps["rms"][ia:ib])
            signal = float(np.clip((peak.get("strength", 0) / 6) + max(0, float(rms.max() - rms.mean()) / 20), 0, 1))
            emotion = {"peak_type": peak.get("type"), "rms": round(float(rms.max()) if len(rms) else 0, 3)}
            cards.append({"id": len(cards), "si": start_i, "ei": end_i, "start": round(a, 3), "end": round(b, 3),
                          "dur": round(duration, 3), "peak": peak.get("id"), "peak_t": t, "peak_type": peak.get("type"),
                          "cluster": f"event{event['id']}_peak{peak.get('id', len(cards))}", "event_id": event["id"],
                          "who": who, "what": what, "emotion": emotion, "object_or_number": _number(nearby),
                          "category": category, "format": fmt, "incident_strength": round(signal, 4),
                          "consensus_likeness": round(_consensus_prior(cfg, category), 4),
                          "viewer_score": round(_comment_score(A, a, b), 4)})
    # Strong shout / scream / laugh peaks with no speech around them (slide rides, screams) fall BETWEEN the
    # sentence-based events above - give each its own card; refine() finds the clean start/end from nearby speech.
    used = {round(c["peak_t"], 3) for c in cards}
    must = cfg["engine"].get("must_peak_strength", 3.0)
    from . import boundaries
    extra = boundaries.excited_moments(A, cfg["engine"].get("excited_min_z", 3.0))   # loud "yaar kya baat hai!" moments
    for peak in list(A.get("peaks", [])) + extra:
        t = float(peak["t"])
        if round(t, 3) in used or peak.get("type") not in ("scream", "laugh", "cheer", "gasp", "excited") or peak.get("strength", 0) < must:
            continue
        if any(abs(t - u) < 10 for u in used):
            continue
        if peak.get("type") != "excited" and any(e["start"] <= t <= e["end"] for e in events):
            continue
        used.add(round(t, 3))
        si = max(0, _sentence_index(sents, t))
        nearby = _text(A, t - 15, t + 15)
        category = _category(nearby, peak.get("type"))
        ia, ib = int(max(0, t - 5)), min(len(ps["rms"]), int(t + 5))
        rms = np.asarray(ps["rms"][ia:ib])
        signal = float(np.clip((peak.get("strength", 0) / 6) + max(0, float(rms.max() - rms.mean()) / 20), 0, 1))
        cards.append({"id": len(cards), "si": si, "ei": si, "start": round(t - 20, 3), "end": round(t + 15, 3), "dur": 35.0,
                      "peak": peak.get("id"), "peak_t": t, "peak_type": peak.get("type"),
                      "cluster": f"gap_peak{peak.get('id')}", "event_id": f"gap{peak.get('id')}",
                      "who": _near_names(nearby), "what": peak.get("type"), "emotion": {"peak_type": peak.get("type")},
                      "object_or_number": None, "category": category, "format": "moment",
                      "incident_strength": round(signal, 4), "consensus_likeness": round(_consensus_prior(cfg, category), 4),
                      "viewer_score": round(_comment_score(A, t - 20, t + 15), 4)})
    # Keep the strongest card from the same event/peak family before scoring.
    unique = {}
    for card in cards:
        key = (card["event_id"], card["peak_t"])
        if key not in unique or card["incident_strength"] > unique[key]["incident_strength"]:
            unique[key] = card
    return list(unique.values())


def _number(text):
    m = re.search(r"(?<!\\w)(?:₹|rs\\.?\\s*)?\\d+(?:[.,]\\d+)?(?:\\s*(?:lakh|crore|rupee))?", hinglish.romanize(text).lower())
    return m.group(0) if m else None


def title(card):
    who = card["who"][0] if card["who"] else ""
    what = card["what"] or ""
    ending = "ho gaya" if card["category"] in ("funny_accident", "danger_loss_fear") else "gaya"
    emoji = EMOJI.get(card["category"], "😂")
    text = f"{who} {what} {ending} {emoji}"
    return " ".join(text.split())[:40].rstrip()


def score_cards(cards):
    for card in cards:
        card["title"] = title(card)
        card["titleable"] = bool(card["who"] and card["what"] and card["title"])
        card["final"] = round(0.30 * card["incident_strength"] + 0.25 * card["consensus_likeness"] +
                              0.20 * card["viewer_score"] + 0.25 * (1 if card["titleable"] else 0), 6)
        card["hook_score"] = card["incident_strength"]
        card["story_score"] = round(0.6 * card["incident_strength"] + 0.4 * card["consensus_likeness"], 6)
        card["hook_types"] = [card["category"], card["peak_type"]]
        card["reasons"] = [f"{card['category']} incident", f"who={card['who'][0] if card['who'] else '-'}", f"what={card['what']}"]
        card["reject"] = None if card["titleable"] else "not_titleable"
        card["reject_all"] = [] if card["titleable"] else ["not_titleable"]
    return cards


def select(cards, A, cfg, max_shorts=None):
    engine = cfg["engine"]
    target_max = min(max_shorts or engine["max_shorts"], engine["max_shorts"])
    target = max(engine["min_shorts"], min(target_max, max(1, round(len(cards) / 2))))
    chosen = []
    skips = []
    for card in sorted((x for x in cards if not x["reject"]), key=lambda x: (-x["final"], x["start"])):
        if len(chosen) >= target:
            break
        if card["final"] < engine["min_score"]:
            card["reject"] = "below_min_score"; card["reject_all"].append("below_min_score"); continue
        if any(_overlap(card, x) > 0.10 for x in chosen):
            skips.append({"cand": card["id"], "skip": "overlap"}); continue
        card["picked_by"] = "moment_score"
        card["tag"] = card["category"]
        chosen.append(card)
    chosen.sort(key=lambda x: x["start"])
    return {"target_count": target, "clusters": {x["cluster"]: {"versions": 1, "best": x["id"]} for x in cards},
            "winners": [x["id"] for x in chosen], "skips": skips}


def _overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"])) / min(a["dur"], b["dur"])


def refine(cards, A, cfg):
    """Deterministic best start/end (silence-cut, ms precise) + a real-line title for every card."""
    from . import boundaries
    out = []
    for card in cards:
        b = boundaries.best_span(A, float(card["peak_t"]), cfg)
        if b is None:
            card["reject"] = "no_clean_span"; card["reject_all"] = ["no_clean_span"]
            out.append(card); continue
        card.update(start=b["start"], end=b["end"], dur=b["dur"], si=b["si"], ei=b["ei"], format="moment",
                    boundary=b)
        card["title"], card["title_source"] = boundaries.make_title(A, card)
        card["reject"], card["reject_all"] = None, []               # a real line / template always exists now
        shout = boundaries.shout_strength(A, card)
        card["shout"] = shout
        card["final"] = round(card["final"] + 0.25 * b["score"] + cfg["boundaries"].get("shout_weight", 0.6) * shout, 6)
        out.append(card)
    return out


def plan(A, cfg, max_shorts=None):
    cards = score_cards(incident_cards(A, cfg))
    if cfg.get("boundaries", {}).get("enabled", False):
        cards = refine(cards, A, cfg)
    sel = select(cards, A, cfg, max_shorts)
    if cfg.get("boundaries", {}).get("enabled", False):     # unique titles, best Short gets first pick
        from . import boundaries
        used = set()
        for cid in sorted(sel["winners"], key=lambda i: (-cards[i]["final"], cards[i]["start"])):
            cards[cid]["title"], cards[cid]["title_source"] = boundaries.make_title(A, cards[cid], used)
            used.add(cards[cid]["title"])
    return cards, sel
