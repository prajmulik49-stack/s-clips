"""Topic boundaries (TextTiling on TF-IDF of phonetic keys + location + ambience shifts) and peak moments."""
from __future__ import annotations

import numpy as np

from ..text import lex


def tfidf_matrix(docs: list[list[str]]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer=lambda d: d, min_df=1, norm="l2", sublinear_tf=True)
    if not any(docs):
        return None, v
    return v.fit_transform(docs), v


def _cos(a, b):
    num = float(a.multiply(b).sum())
    na = float(np.sqrt(a.multiply(a).sum())); nb = float(np.sqrt(b.multiply(b).sum()))
    return num / (na * nb) if na > 0 and nb > 0 else 0.0


def bhattacharyya(h1, h2) -> float:
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(h1 * h2)))))


def location_curve(hist: np.ndarray, vfps: float, n_sec: int, win_s: float) -> np.ndarray:
    """Per-second colour-distribution shift between the win_s before and after each second."""
    out = np.zeros(n_sec, np.float32)
    w = max(1, int(win_s * vfps))
    for t in range(n_sec):
        i = int(t * vfps)
        if i - w < 0 or i + w > len(hist):
            continue
        out[t] = bhattacharyya(hist[i - w:i].mean(0), hist[i:i + w].mean(0))
    return out


def boundaries(sents: list[dict], words: list[dict], per_sec: dict, loc: np.ndarray, cfg_t: dict) -> dict:
    k = cfg_t["block_sentences"]
    docs = [lex.content_keys(s["text"]) for s in sents]
    X, _ = tfidf_matrix(docs)
    n = len(sents)
    sim = np.ones(n, np.float32)
    if X is not None:
        for i in range(1, n):
            L = X[max(0, i - k):i].sum(0); R = X[i:min(n, i + k)].sum(0)
            from scipy.sparse import csr_matrix
            sim[i] = _cos(csr_matrix(L), csr_matrix(R))
    depth = np.zeros(n, np.float32)
    for i in range(1, n - 1):
        lp = sim[max(1, i - k):i + 1].max(); rp = sim[i:min(n, i + k + 1)].max()
        depth[i] = (lp - sim[i]) + (rp - sim[i])
    text_sc = np.clip(depth / max(1e-6, 2 * cfg_t["boundary_min_depth"] * 2), 0, 1)
    amb = per_sec["ambience_vec"]
    amb_sc = np.zeros(n, np.float32); loc_sc = np.zeros(n, np.float32)
    win = int(cfg_t["location_hist_window_s"])
    for i, s in enumerate(sents):
        t = int(s["s"])
        if win <= t < len(amb) - win:
            d = np.abs(amb[t - win:t].mean(0) - amb[t:t + win].mean(0)).sum()
            amb_sc[i] = float(np.clip(d / cfg_t["ambience_threshold"], 0, 1))
        lo = loc[max(0, t - 2):t + 3]
        loc_sc[i] = float(np.clip((lo.max() if len(lo) else 0) / cfg_t["location_hist_threshold"], 0, 1))
    w = cfg_t["weights"]
    score = w["text"] * text_sc + w["location"] * loc_sc + w["ambience"] * amb_sc
    pause = np.array([s["pause_before"] for s in sents])
    score = score + cfg_t["pause_bonus"] * np.clip((pause - 1.0) / 3.0, 0, 1)   # long pause = mild boundary evidence
    is_b = np.zeros(n, bool)
    if n:
        is_b[0] = True
    gap = cfg_t.get("min_gap_s", 0)
    for i in sorted(range(n), key=lambda i: (-round(float(score[i]), 6), i)):   # NMS: strongest boundary per gap
        if score[i] < cfg_t["boundary_threshold"]:
            break
        if all(abs(sents[i]["s"] - sents[j]["s"]) >= gap for j in np.flatnonzero(is_b)):
            is_b[i] = True
    seg_id, cur = [], 0
    for i in range(n):
        if i > 0 and is_b[i]:
            cur += 1
        seg_id.append(cur)
    # top content keys per topic segment (for the "setup" feature)
    seg_keys = {}
    for sid in sorted(set(seg_id)):
        bag = [kk for i in range(n) if seg_id[i] == sid for kk in docs[i]]
        cnt = {}
        for kk in bag:
            cnt[kk] = cnt.get(kk, 0) + 1
        seg_keys[sid] = [kk for kk, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if c >= 2][:8]
    return {"text_sim": [round(float(x), 4) for x in sim], "boundary_score": [round(float(x), 4) for x in score],
            "location_score": [round(float(x), 4) for x in loc_sc], "is_boundary": [bool(x) for x in is_b],
            "topic_id": seg_id, "topic_keys": {str(k2): v for k2, v in seg_keys.items()}}


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=float(np.nanmedian(x)) if np.any(~np.isnan(x)) else 0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-6:
        mad = x.std() + 1e-6
    return ((x - med) / mad).astype(np.float32)


def peaks(per_sec: dict, words: list[dict], comments: np.ndarray | None, cfg_p: dict) -> list[dict]:
    n = len(per_sec["rms"])
    sig = {"laugh": robust_z(per_sec["laugh"]), "scream": robust_z(per_sec["scream"]),
           "cheer": robust_z(per_sec["cheer"]), "gasp": robust_z(per_sec["gasp"]),
           "energy": robust_z(per_sec["rms"])}
    rv = np.zeros(n, np.float32)
    rev = {lex.key_word(w) for p in lex.lexicon().get("hook", {}).get("reveal", []) for w in p.split()[:1]}
    for w in words:
        if lex.key_word(w["w"]) in rev and int(w["s"]) < n:
            rv[int(w["s"])] = 3.0
    sig["reveal"] = rv
    if comments is not None and len(comments) == n and comments.sum() > 0:
        sig["comments"] = robust_z(comments)
    wts = cfg_p["weights"]
    comb = np.zeros(n, np.float32)
    contrib = np.zeros((len(sig), n), np.float32)
    names = list(sig)
    for j, k in enumerate(names):
        c = wts.get(k, 0.0) * np.clip(sig[k], 0, 6)
        contrib[j] = c; comb += c
    # smooth over 3 s, pick local maxima with non-maximum suppression
    sm = np.convolve(comb, np.ones(3) / 3, mode="same")
    order = sorted(range(n), key=lambda i: (-round(float(sm[i]), 6), i))
    taken, out = np.zeros(n, bool), []
    sep = int(cfg_p["min_separation_s"])
    for i in order:
        if taken[i]:
            continue
        best_sig = max(sig[k][i] for k in names)
        if best_sig < cfg_p["min_z"] or sm[i] <= 0:
            continue
        lo, hi = max(0, i - 1), min(n, i + 2)
        t = int(lo + np.argmax(comb[lo:hi]))
        j = int(np.argmax(contrib[:, t]))
        out.append({"id": len(out), "t": float(t), "strength": round(float(sm[i]), 6), "type": names[j],
                    "signals": {k: round(float(sig[k][t]), 3) for k in names}})
        taken[max(0, i - sep):i + sep + 1] = True
    out.sort(key=lambda p: p["t"])
    for k2, p in enumerate(out):
        p["id"] = k2
    return out
