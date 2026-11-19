"""Hook (first 3 s) and story features + hand-weighted fallback blend. Every feature is in [0, 1].

final = 0.40*hook + 0.45*story + 0.15*viewer   (weights from config; a missing signal's weight is redistributed)
Scores are rounded to 6 decimals; ties -> earliest start.
"""
from __future__ import annotations

import numpy as np

from .analysis.topics import robust_z, tfidf_matrix
from .text import lex

LABELS = {
    "first_word": "speech starts instantly", "first_sentence": "short, complete first line",
    "energy_jump": "loud opening", "pitch_jump": "animated voice at the start", "question": "opens with a question",
    "reveal": "reveal cue in the hook", "danger": "danger/fear in the hook", "conflict": "argument / fast turns at start",
    "funny": "laugh in the first 3 s", "name_call": "family name called out", "emotional": "emotional opening",
    "open_loop": "hook pays off later", "lexicon": "Sourav catchphrase / reaction words",
    "setup": "sets up the topic early", "arc": "rise-peak-release arc", "peak_position": "peak lands at 55-85%",
    "payoff": "reaction right after the peak", "one_scene": "single scene", "coherence": "stays on one topic",
    "interaction": "back-and-forth between people", "loop": "ending loops to the start", "dead_air": "dead air",
    "viewer": "viewers timestamped this moment",
}


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


class Context:
    """Per-video precomputation shared by all candidates."""

    def __init__(self, A: dict, cfg):
        self.A, self.cfg = A, cfg
        ps = A["per_sec"]
        self.ps = ps
        self.z = {k: robust_z(ps[k]) for k in ("rms", "laugh", "scream", "cheer", "gasp", "pitch_std")}
        inten = 0.5 * self.z["rms"] + np.clip(self.z["laugh"], 0, 6) + 0.8 * np.clip(self.z["scream"], 0, 6) \
            + 0.5 * np.clip(self.z["cheer"], 0, 6)
        self.intensity = np.convolve(inten, np.ones(5) / 5, mode="same")
        docs = [lex.content_keys(s["text"]) for s in A["sentences"]]
        self.docs = docs
        X, _ = tfidf_matrix(docs)
        self.X = X
        self.comments = A.get("comments_per_sec")
        self.pitch = ps["pitch_med"]

    def sim(self, i, j) -> float:
        if self.X is None:
            return 0.0
        a, b = self.X[i], self.X[j]
        return float(a.multiply(b).sum())   # rows are l2-normalized


def _words(A, a, b):
    return [w for w in A["words"] if a <= w["s"] < b]


def _text(ws):
    return " ".join(w["w"] for w in ws)


def hook_features(ctx: Context, c: dict) -> tuple[dict, list[str]]:
    A, cfg, ps, z = ctx.A, ctx.cfg, ctx.ps, ctx.z
    sc = cfg["scoring"]
    H = sc["hook_window_s"]
    a, b = c["start"], c["end"]
    ia, ib, ih = int(a), int(np.ceil(b)), int(a + H)
    sents = A["sentences"]
    s0 = sents[c["si"]]
    ws = _words(A, a, a + H)
    txt3 = _text(ws)
    first_text = s0["text"]
    f = {}
    t_first = (ws[0]["s"] - a) if ws else H
    f["first_word"] = _clip01((sc["first_word_worst_s"] - t_first) / (sc["first_word_worst_s"] - sc["first_word_best_s"]))
    d0 = s0["e"] - s0["s"]
    f["first_sentence"] = (1.0 if d0 <= 4 else 0.5 if d0 <= 6 else 0.1) * (1.0 if s0["complete"] else 0.4)
    clip_rms = ps["rms"][ia:ib]
    f["energy_jump"] = _clip01(0.5 + (ps["rms"][ia:ih + 1].mean() - clip_rms.mean()) / (2 * (clip_rms.std() + 1e-6)))
    clip_p = ps["pitch_std"][ia:ib]
    f["pitch_jump"] = _clip01(0.5 + (ps["pitch_std"][ia:ih + 1].mean() - clip_p.mean()) / (2 * (clip_p.std() + 1e-6)))
    r = cfg["filters"]["phrase_match_ratio"]
    hook_lex = lex.lexicon().get("hook", {})

    def has(cat, text):
        return any(lex.phrase_in(p, text, r) for p in hook_lex.get(cat, []))

    # rising pitch at the end of the first sentence
    pm = ctx.pitch
    e0 = int(min(s0["e"], a + H))
    tail = pm[max(ia, e0 - 1):e0 + 1]
    rising = bool(len(tail) == 2 and not np.isnan(tail).any() and tail[1] - tail[0] > 1.5)
    f["question"] = 1.0 if ("?" in first_text or has("question", first_text)) else (0.5 if rising else 0.0)
    f["reveal"] = 1.0 if has("reveal", txt3) else (0.6 if z["gasp"][ia:ih + 1].max() > 2 else 0.0)
    f["danger"] = 1.0 if has("danger", txt3) else (0.7 if z["scream"][ia:ih + 1].max() > 2.5 else 0.0)
    spk = [s["spk"] for s in sents if s["s"] < a + H and s["e"] > a]
    turns = sum(1 for x, y in zip(spk, spk[1:]) if x != y)
    f["conflict"] = _clip01(0.5 * turns) * (1.0 if z["rms"][ia:ih + 1].mean() > 0.5 else 0.5)
    f["funny"] = _clip01((z["laugh"][ia:ih + 1].max() - 1.0) / 2.0)
    names = lex.lexicon().get("names", [])
    name_hit = any(lex.phrase_in(n, txt3, 100) for n in names)
    f["name_call"] = (1.0 if z["rms"][ia:ih + 1].max() > 1.0 else 0.4) if name_hit else 0.0
    slow = ps["words"][ia:ih + 1].mean() < max(1.0, ps["words"][ia:ib].mean() * 0.8)
    f["emotional"] = (1.0 if slow else 0.5) if has("emotional", txt3) else 0.0
    hk = set(lex.content_keys(first_text))
    rest = lex.content_keys(_text(_words(A, s0["e"], b)))
    f["open_loop"] = _clip01(sum(1 for k in set(rest) if k in hk) / 2.0)
    f["lexicon"] = _clip01(len(lex.hits("reactions", txt3, r)) / 2.0)
    types = [k for k in ("question", "reveal", "danger", "conflict", "funny", "name_call", "emotional") if f[k] >= 0.5]
    return f, types


def story_features(ctx: Context, c: dict) -> dict:
    A, cfg, ps, z = ctx.A, ctx.cfg, ctx.ps, ctx.z
    sc = cfg["scoring"]
    a, b = c["start"], c["end"]
    ia, ib = int(a), int(np.ceil(b))
    sents = A["sentences"]
    f = {}
    topic = sents[c["si"]]["topic"]
    tkeys = set(A["topics"]["topic_keys"].get(str(topic), []))
    early = set(lex.content_keys(_text(_words(A, a, a + 10))))
    f["setup"] = 1.0 if tkeys & early else (0.3 if early else 0.0)
    seg = ctx.intensity[ia:ib]
    if len(seg) >= 8 and seg.std() > 1e-6:
        x = np.interp(np.linspace(0, len(seg) - 1, 20), np.arange(len(seg)), seg)
        tpl = np.concatenate([np.linspace(0.2, 1.0, 14), np.linspace(0.9, 0.4, 6)])
        f["arc"] = _clip01((np.corrcoef(x, tpl)[0, 1] + 1) / 2)
    else:
        f["arc"] = 0.3
    lo, hi = sc["peak_pos_range"]
    if c["peak_t"] is not None:
        pos = (c["peak_t"] - a) / (b - a)
        dist = max(0.0, lo - pos, pos - hi)
        f["peak_position"] = _clip01(1 - dist / 0.3)
        pt = int(c["peak_t"])
        after = slice(pt + 1, min(ib, pt + 1 + int(sc["payoff_after_peak_s"])))
        react = np.maximum(z["laugh"][after], z["cheer"][after]) if after.stop > after.start else np.array([0.0])
        pay = 0.0
        if len(react) and react.max() > 1.5:
            t_pay = pt + 1 + int(np.argmax(react))
            pay = 0.6 + (0.4 if b - t_pay <= sc["end_after_payoff_s"] else 0.15)
        elif c["peak_type"] in ("laugh", "cheer"):
            pay = 0.5 + (0.3 if b - c["peak_t"] <= sc["end_after_payoff_s"] + 4 else 0.0)
        f["payoff"] = _clip01(pay)
    else:
        f["peak_position"] = 0.0; f["payoff"] = 0.0
    cuts = ps["scene_cut"][ia:ib].sum()
    f["one_scene"] = _clip01(1 - cuts / 12.0)
    idx = list(range(c["si"], c["ei"] + 1))
    sims = [ctx.sim(i, j) for i, j in zip(idx, idx[1:]) if ctx.docs[i] and ctx.docs[j]]
    f["coherence"] = _clip01((np.mean(sims) if sims else 0.0) * 3)
    spk = [sents[i]["spk"] for i in idx]
    turns = sum(1 for x, y in zip(spk, spk[1:]) if x != y)
    abab = sum(1 for k in range(2, len(spk)) if spk[k] == spk[k - 2] != spk[k - 1])
    faces = ps["faces"][ia:ib]
    f["interaction"] = _clip01(0.5 * min(1, turns / max(1, (b - a) / 60 * 8)) + 0.3 * min(1, faces.mean() / 1.5)
                               + 0.2 * min(1, abab / 3))
    gaps = 0.0
    prev_e = a
    for w in _words(A, a, b):
        g = w["s"] - prev_e
        if g > sc["dead_air_gap_s"] and ps["laugh"][int(prev_e):int(w["s"]) + 1].max() < 0.3:
            gaps += g
        prev_e = max(prev_e, w["e"])
    if b - prev_e > sc["dead_air_gap_s"]:
        gaps += b - prev_e
    f["dead_air"] = _clip01(gaps / (b - a) * 4)
    first = set(lex.content_keys(sents[c["si"]]["text"]))
    last = set(lex.content_keys(sents[c["ei"]]["text"]))
    f["loop"] = _clip01(len(first & last) / 2.0)
    return f


def viewer_feature(ctx: Context, c: dict) -> float | None:
    cm = ctx.comments
    if cm is None or cm.sum() <= 0:
        return None
    a, b = int(c["start"]), int(np.ceil(c["end"]))
    dens = cm[a:b].sum() / max(1.0, (b - a))
    ref = np.percentile(np.convolve(cm, np.ones(45), "valid") / 45, 95) + 1e-9
    return _clip01(dens / ref)


def score_all(A: dict, cands: list[dict], cfg) -> None:
    ctx = Context(A, cfg)
    sc = cfg["scoring"]
    hw, sw = sc["hook"], sc["story"]
    R = sc["round"]
    for c in cands:
        hf, types = hook_features(ctx, c)
        sf = story_features(ctx, c)
        v = viewer_feature(ctx, c)
        hook = sum(hw[k] * hf[k] for k in hf) / sum(hw[k] for k in hf)
        pos = [k for k in sf if k != "dead_air"]
        story = (sum(sw[k] * sf[k] for k in pos) - sw["dead_air"] * sf["dead_air"]) / sum(sw[k] for k in pos)
        story = _clip01(story)
        wf = dict(sc["weights_final"])
        parts = {"hook": hook, "story": story, "viewer": v}
        avail = {k: w for k, w in wf.items() if parts[k] is not None}
        tot = sum(avail.values())
        final = sum(w * parts[k] for k, w in avail.items()) / tot
        c["hook_features"] = {k: round(x, R) for k, x in hf.items()}
        c["story_features"] = {k: round(x, R) for k, x in sf.items()}
        c["hook_types"] = types
        c["hook_score"] = round(hook, R)
        c["story_score"] = round(story, R)
        c["viewer_score"] = None if v is None else round(v, R)
        c["final"] = round(final, R)
        c["blend"] = {k: round(w / tot, 4) for k, w in avail.items()}
        contrib = {("hook", k): hw[k] * hf[k] / sum(hw.values()) * avail["hook"] / tot for k in hf}
        contrib |= {("story", k): sw[k] * sf[k] / sum(sw[k] for k in pos) * avail["story"] / tot for k in pos}
        if v is not None:
            contrib[("viewer", "viewer")] = v * avail["viewer"] / tot
        top = sorted(contrib.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))[:3]
        c["reasons"] = [f"{LABELS[k]} ({part} {(hf if part == 'hook' else sf).get(k, v) if part != 'viewer' else v:.2f})"
                        for (part, k), _ in top]
