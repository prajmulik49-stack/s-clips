"""Deterministic best start / end for a moment, cut inside silence at millisecond precision, plus quote titles.

For a moment peak t:
  1. every (sentence start S, sentence end E) pair around t whose cut length is within [min_s, max_s] is scored:
       start quality  - real pause before, topic/scene boundary, no dependent first word ("aur", "toh", ...)
       end quality    - complete sentence, pause after, the reaction/payoff after the peak is inside, no dangling word
       peak position  - the peak sits at 40-80 % of the clip
       length         - inside the preferred range (default 38-50 s)
       dead air       - long silences inside are penalised
     best total wins; ties -> earliest start (deterministic).
  2. cut points: the quietest 10 ms frame inside the silence before the first word / after the last word,
     never inside a word (later snapped to the frame grid by the renderer).
"""
from __future__ import annotations

import re

import numpy as np

from .text import hinglish, lex

EMOJIS = {"funny_accident": "😂🤣", "danger_loss_fear": "😱😢", "festival_emotional": "🥺❤️", "drama_relationship": "😤😳",
          "dance": "💃🔥", "money_numbers": "💰😳", "travel_scenery": "😍🌄", "food": "😋"}


def _quiet_point(rms10: np.ndarray, a: float, b: float, default: float) -> float:
    """Time of the lowest-energy 10 ms frame in [a, b] (falls back to `default` if the window is empty)."""
    ia, ib = int(np.ceil(a * 100)), int(np.floor(b * 100))
    if ib <= ia or ia < 0 or ib > len(rms10):
        return default
    seg = rms10[ia:ib + 1]
    k = int(np.argmin(np.round(seg, 2)))          # rounding keeps the choice stable; argmin -> earliest on ties
    return (ia + k) / 100.0


def _lin(x, lo, hi):
    return float(np.clip((x - lo) / max(1e-9, hi - lo), 0, 1))


def best_span(A: dict, t: float, cfg: dict, lo_bound: float = 0.0, hi_bound: float | None = None) -> dict | None:
    c = cfg["boundaries"]
    sents, words, ps = A["sentences"], A["words"], A["per_sec"]
    rms10 = A["rms10"]
    hi_bound = A["duration"] if hi_bound is None else hi_bound
    z_laugh = ps["laugh"]; z_cheer = ps["cheer"]
    starts = [s for s in sents if t - c["max_lead_s"] <= s["s"] <= t - c["min_lead_s"] and s["s"] >= lo_bound]
    ends = [s for s in sents if t + c["min_tail_s"] <= s["e"] <= t + c["max_tail_s"] and s["e"] <= hi_bound]
    best = None
    for S in starts:
        w0 = words[S["w0"]]
        prev_e = words[S["w0"] - 1]["e"] if S["w0"] > 0 else max(0.0, w0["s"] - 1.0)
        a = _quiet_point(rms10, max(prev_e + 0.02, w0["s"] - c["max_pre_roll_s"]), w0["s"] - 0.03, w0["s"] - 0.05)
        dep = lex.first_word_dependent(S["text"])
        intro = lex.hits("intro_outro", S["text"]) or lex.hits("sponsor", S["text"])
        sq = (0.35 * _lin(S["pause_before"], 0.1, 1.2) + 0.25 * (0 if dep else 1)
              + 0.2 * (1 if S.get("is_boundary") else 0.4 * _lin(S.get("boundary_score", 0), 0.1, 0.6))
              + 0.2 * (1 if (S["e"] - S["s"]) <= 5 and S["complete"] else 0.4))
        if intro:
            continue
        for E in ends:
            if E["id"] < S["id"]:
                continue
            wl = words[E["w1"] - 1]
            nxt_s = words[E["w1"]]["s"] if E["w1"] < len(words) else wl["e"] + 1.0
            b = _quiet_point(rms10, wl["e"] + 0.03, min(nxt_s - 0.02, wl["e"] + c["max_post_roll_s"]), wl["e"] + 0.08)
            dur = b - a
            if not (c["min_s"] <= dur <= c["max_s"]):
                continue
            ia, ib = int(t), int(np.ceil(b))
            payoff = float(max(z_laugh[ia:ib].max(initial=0), z_cheer[ia:ib].max(initial=0)) >= 0.3)
            eq = (0.35 * (1 if E["complete"] else 0) + 0.25 * _lin(E["pause_after"], 0.1, 1.0)
                  + 0.2 * payoff + 0.2 * (1 if E["punct_end"] or E["pause_after"] >= 0.6 else 0))
            pos = (t - a) / dur
            pq = 1.0 - _lin(abs(pos - 0.6), 0.2, 0.45)
            lq = 1.0 if c["pref_s"][0] <= dur <= c["pref_s"][1] else (
                _lin(dur, c["min_s"], c["pref_s"][0]) if dur < c["pref_s"][0] else 1 - _lin(dur, c["pref_s"][1], c["max_s"]))
            gaps = 0.0
            prev = a
            for w in words[S["w0"]:E["w1"]]:
                if w["s"] - prev > 1.5:
                    gaps += w["s"] - prev
                prev = max(prev, w["e"])
            dq = _lin(gaps / dur, 0.05, 0.35)
            total = round(0.30 * sq + 0.30 * eq + 0.20 * pq + 0.20 * lq - 0.20 * dq, 6)
            key = (total, -a)
            if best is None or key > best["_key"]:
                best = {"_key": key, "start": round(a, 3), "end": round(b, 3), "dur": round(dur, 3), "si": S["id"],
                        "ei": E["id"], "score": total, "parts": {"start": round(sq, 3), "end": round(eq, 3),
                        "peak_pos": round(pq, 3), "length": round(lq, 3), "dead_air": round(dq, 3)},
                        "first_words": " ".join(hinglish.romanize(w["w"]) for w in words[S["w0"]:S["w0"] + 6]),
                        "last_words": " ".join(hinglish.romanize(w["w"]) for w in words[max(S["w0"], E["w1"] - 6):E["w1"]])}
    if best:
        best.pop("_key")
    return best


# ---------------------------------------------------------------- titles: a real line from the clip
def quote_title(A: dict, card: dict, max_chars: int = 38) -> str | None:
    sents = A["sentences"][card["si"]:card["ei"] + 1]
    words = A["words"]
    t = card.get("peak_t") or (card["start"] + card["end"]) / 2
    ps = A["per_sec"]
    names = {lex.key_word(n): n for n in (lex.lexicon().get("names") or [])}
    best, bkey = None, None
    for s in sents:
        ws = words[s["w0"]:s["w1"]]
        if not ws or np.mean([w.get("p", 0.9) for w in ws]) < 0.55:
            continue
        rom = [re.sub(r"[^A-Za-z0-9?!']", "", hinglish.romanize(w["w"])) for w in ws]
        rom = [r for i, r in enumerate(rom) if r and (i == 0 or r.lower() != rom[i - 1].lower())]   # drop stutters
        stop = {lex.key_word(x) for x in (lex.lexicon().get("stopwords") or [])} |                {lex.key_word(x) for x in ("se", "aur", "ki", "ke", "ka", "ko", "mein", "to", "toh", "ba", "par", "ye", "vo", "na")}
        while rom and lex.key_word(rom[0].strip("?!")) in stop:     # no leading / trailing filler words
            rom = rom[1:]
        while rom and lex.key_word(rom[-1].strip("?!")) in stop:
            rom = rom[:-1]
        content = [r for r in rom if lex.key_word(r.strip("?!")) not in stop and len(r) >= 3]
        if not (3 <= len(rom) <= 8) or len(content) < 2:
            continue
        text = " ".join(rom)
        if len(text) > max_chars or lex.first_word_dependent(s["text"]) or lex.hits("intro_outro", s["text"]):
            continue
        has_name = any(lex.key_word(r) in names for r in rom)
        energy = float(ps["rms"][int(s["s"]):int(s["e"]) + 1].mean())
        dist = abs((s["s"] + s["e"]) / 2 - t)
        k = (round(-dist / 8 + energy / 25 + (0.5 if s["complete"] else 0) + (0.4 if has_name else 0)
                   + (0.3 if "?" in text or "!" in text else 0), 6), -s["s"])
        if bkey is None or k > bkey:
            best, bkey = text, k
    if not best:
        return None
    out = []
    for w in best.split():
        k = lex.key_word(w)
        out.append(names.get(k, w[:1].upper() + w[1:].lower()))
    return " ".join(out) + " " + EMOJIS.get(card.get("category"), "😂🤣")


QUESTION_WORDS = ("kya", "kitna", "kitni", "kitne", "kaise", "kaisa", "kaun", "kahan", "kyun", "kyon", "kab", "kidhar")
TEMPLATES = {"danger_loss_fear": ["Itna Darr Laga Sabko", "Bach Gaye Aaj Toh", "Dil Dhak Dhak Karne Laga", "Ye Kya Ho Gaya"],
             "funny_accident": ["Sab Hans Hans Ke Pagal", "Hasi Nahi Ruk Rahi", "Ye Kya Kar Diya", "Pagalpan Ho Gaya"],
             "travel_scenery": ["Ye Jagah Toh Dekho", "Kya View Hai Yaar", "Aisi Jagah Pehli Baar Dekhi"],
             "festival_emotional": ["Ye Pal Kabhi Nahi Bhoolenge", "Dil Khush Ho Gaya"],
             "drama_relationship": ["Ab Kya Hoga", "Ye Toh Hona Hi Tha"], "dance": ["Kya Dance Kiya", "Sab Nachne Lage"],
             "money_numbers": ["Itna Mehenga", "Paise Hi Paise"]}


def _is_question(text_rom: str, raw: str) -> bool:
    if "?" in raw or "?" in text_rom:
        return True
    toks = [lex.key_word(t) for t in text_rom.lower().split()]
    qk = {lex.key_word(q) for q in QUESTION_WORDS}
    return any(t in qk for t in toks)


def question_title(A: dict, card: dict, max_chars: int = 38) -> str | None:
    """A question someone asks in the clip ("Arre ye kya hai?", "Ye kitna lamba hai?") - closest to the peak wins."""
    words = A["words"]
    t = card.get("peak_t") or (card["start"] + card["end"]) / 2
    best, bkey = None, None
    for s in A["sentences"][card["si"]:card["ei"] + 1]:
        ws = words[s["w0"]:s["w1"]]
        if not ws or np.mean([w.get("p", 0.9) for w in ws]) < 0.5:
            continue
        rom = [re.sub(r"[^A-Za-z0-9?!']", "", hinglish.romanize(w["w"])) for w in ws]
        rom = [r for i, r in enumerate(rom) if r and (i == 0 or r.lower() != rom[i - 1].lower())]
        lead = {lex.key_word(x) for x in ("aur", "to", "toh", "phir", "fir", "ki", "se", "ke", "na", "ab")}
        while rom and lex.key_word(rom[0].strip("?!")) in lead:       # "Aur kaisa?" -> "Kaisa?"
            rom = rom[1:]
        if not (3 <= len(rom) <= 8):                                    # "Kya mast?" is too thin for a title
            continue
        text = " ".join(rom)
        if len(text) > max_chars - 1 or not _is_question(text, s["text"]):
            continue
        k = (round(-abs((s["s"] + s["e"]) / 2 - t) / 6 + (0.5 if "?" in s["text"] else 0), 6), -s["s"])
        if bkey is None or k > bkey:
            best, bkey = text.rstrip("?!") + "?", k
    if not best:
        return None
    names = {lex.key_word(n): n for n in (lex.lexicon().get("names") or [])}
    out = [names.get(lex.key_word(w.strip("?")), w[:1].upper() + w[1:].lower()) for w in best.split()]
    return " ".join(out) + " " + EMOJIS.get(card.get("category"), "😱😂")


def make_title(A: dict, card: dict, used: set | None = None) -> tuple[str, str]:
    """Priority: a question asked in the clip -> the clearest real line -> a fixed template by moment type.
    `used` titles are skipped so one video never repeats a title."""
    used = used if used is not None else set()
    for fn, src in ((question_title, "question"), (quote_title, "quote")):
        q = fn(A, card)
        if q and q not in used:
            return q, src
    cat = card.get("category") or "funny_accident"
    pool = TEMPLATES.get(cat, ["Ye Dekho"]) + [x for k in sorted(TEMPLATES) if k != cat for x in TEMPLATES[k]]
    for tpl in pool:
        t = f"{tpl} {EMOJIS.get(cat, '😂🤣')}"
        if t not in used:
            return t, "template"
    return f"{pool[0]} {EMOJIS.get(cat, '😂🤣')}", "template"


def shout_strength(A: dict, card: dict) -> float:
    """0..1: how strong the loudest shout / scream / laugh inside the clip is (robust z, per video)."""
    from .analysis.topics import robust_z
    ps = A["per_sec"]
    if "_shout_z" not in A:
        A["_shout_z"] = np.maximum(np.maximum(robust_z(ps["scream"]), 0.7 * robust_z(ps["laugh"])), excited_z(A))
    z = A["_shout_z"][int(card["start"]):int(np.ceil(card["end"]))]
    return float(np.clip((z.max(initial=0) - 2.0) / 8.0, 0, 1))


def excited_z(A: dict) -> np.ndarray:
    """Per-second 'loud excited speech' score: someone (or several people) talking much louder than this vlog's
    normal level, with an animated pitch; boosted when an excitement phrase ("yaar kya baat hai", "kitna badiya")
    is said in that second. Robust z units, comparable with the scream / laugh z-scores."""
    from .analysis.topics import robust_z
    if "_excited_z" in A:
        return A["_excited_z"]
    ps = A["per_sec"]
    speech = ps["speech"] >= 0.5
    loud = robust_z(ps["rms_max"])
    pitch = robust_z(ps["pitch_std"])
    z = np.where(speech, 0.8 * loud + 0.4 * np.clip(pitch, 0, None), 0.0)
    for s in A["sentences"]:
        if lex.hits("excite", s["text"]):
            a, b = int(s["s"]), int(np.ceil(s["e"]))
            z[a:b] = np.maximum(z[a:b], 2.0 + np.clip(loud[a:b], 0, None))   # an excitement phrase said loudly
    A["_excited_z"] = z.astype(np.float32)
    return A["_excited_z"]


def excited_moments(A: dict, min_z: float = 3.0, sep_s: float = 20.0) -> list[dict]:
    """Strongest loud-excited-speech seconds (non-max suppressed) -> pseudo peaks for moment cards."""
    z = excited_z(A)
    out, taken = [], np.zeros(len(z), bool)
    for t in sorted(range(len(z)), key=lambda i: (-round(float(z[i]), 4), i)):
        if z[t] < min_z:
            break
        if taken[t]:
            continue
        out.append({"id": f"x{t}", "t": float(t), "type": "excited", "strength": round(float(z[t]), 3)})
        taken[max(0, int(t - sep_s)):int(t + sep_s)] = True
    return out
