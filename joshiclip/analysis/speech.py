"""Words -> refined boundaries (MMS forced alignment + VAD) -> sentences -> speaker turns."""
from __future__ import annotations

import re

import numpy as np

from ..text import hinglish, lex

SR = 16000


# ---------------------------------------------------------------- forced alignment
def _align_chars(word: str) -> str:
    r = hinglish.romanize(word).lower()
    return re.sub(r"[^a-z']", "", r)


def mms_align(x: np.ndarray, segments: list[dict], max_shift: float, log=print) -> tuple[list[dict], dict]:
    """Refine word start/end with torchaudio's MMS_FA (CTC forced alignment on romanized text).
    Each Whisper segment is aligned independently (WhisperX-style). Edges move at most `max_shift`."""
    import torch
    import torchaudio
    torch.manual_seed(0)
    torch.set_num_threads(8)
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).eval()
    dic = bundle.get_dict(star=None)
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    n_ok = n_words = 0
    shifts = []
    out = []
    for seg in segments:
        words = [dict(w) for w in seg["words"]]
        n_words += len(words)
        toks = [_align_chars(w["w"]) for w in words]
        idx = [i for i, t in enumerate(toks) if t and all(c in dic for c in t)]
        a = max(0.0, seg["s"] - 0.3)
        b = seg["e"] + 0.3
        clip = torch.from_numpy(x[int(a * SR):int(b * SR)].copy())[None]
        if not idx or clip.shape[1] < SR * 0.3:
            out.append({**seg, "words": words}); continue
        try:
            with torch.inference_mode():
                em, _ = model(clip)
                spans = aligner(em[0], tokenizer([toks[i] for i in idx]))
            ratio = clip.shape[1] / em.shape[1] / SR
            for i, sp in zip(idx, spans):
                s_new = a + sp[0].start * ratio
                e_new = a + sp[-1].end * ratio
                w = words[i]
                ds, de = s_new - w["s"], e_new - w["e"]
                shifts.append(abs(ds)); shifts.append(abs(de))
                w["s"] = round(w["s"] + float(np.clip(ds, -max_shift, max_shift)), 3)
                w["e"] = round(max(w["s"] + 0.03, w["e"] + float(np.clip(de, -max_shift, max_shift))), 3)
                w["aligned"] = True
                n_ok += 1
        except Exception as e:  # noqa: BLE001  (a segment that cannot be aligned keeps Whisper times)
            log(f"    align skip {seg['s']:.1f}s: {type(e).__name__}")
        out.append({**seg, "words": words})
    stats = {"engine": "torchaudio.MMS_FA", "words": n_words, "aligned": n_ok,
             "median_shift_ms": round(float(np.median(shifts)) * 1000, 1) if shifts else 0.0}
    return out, stats


def vad_snap(words: list[dict], vad_prob: np.ndarray, thr: float, hop: float = 0.032, max_move: float = 0.15) -> int:
    """Move word starts out of non-speech (forward) and ends back into speech, bounded by max_move."""
    n = 0
    sp = vad_prob >= thr
    for w in words:
        i = int(w["s"] / hop)
        j = i
        while j < len(sp) and not sp[j] and (j - i) * hop < max_move:
            j += 1
        if j < len(sp) and sp[j] and j > i and j * hop < w["e"] - 0.03:
            w["s"] = round(j * hop, 3); n += 1
        k = int(w["e"] / hop)
        m = k
        while m > 0 and not sp[min(m, len(sp) - 1)] and (k - m) * hop < max_move:
            m -= 1
        if m < k and (m + 1) * hop > w["s"] + 0.03 and sp[min(m, len(sp) - 1)]:
            w["e"] = round((m + 1) * hop, 3); n += 1
    return n


# ---------------------------------------------------------------- sentences
def build_sentences(words: list[dict], cfg_s: dict) -> list[dict]:
    end_p = tuple(cfg_s["end_punct"])
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - w["e"]) if nxt else 9.0
        seg_end = w.get("seg_last") and gap >= cfg_s.get("segment_split_pause_s", 0.2)
        if w["w"].endswith(end_p) or gap >= cfg_s["split_pause_s"] or seg_end:
            groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    # split run-ons at the largest internal gap
    out = []
    stack = list(reversed(groups))
    while stack:
        g = stack.pop()
        if words[g[-1]]["e"] - words[g[0]]["s"] > cfg_s["max_len_s"] and len(g) > 3:
            gaps = [words[g[k + 1]]["s"] - words[g[k]]["e"] for k in range(len(g) - 1)]
            k = int(np.argmax(gaps[1:-1])) + 1 if len(gaps) > 2 else 0
            stack += [g[k + 1:], g[:k + 1]]
        else:
            out.append(g)
    sents = []
    for g in out:
        ws = [words[i] for i in g]
        text = " ".join(w["w"] for w in ws)
        nxt = words[g[-1] + 1] if g[-1] + 1 < len(words) else None
        pause_after = round((nxt["s"] - ws[-1]["e"]) if nxt else 9.0, 3)
        prv = words[g[0] - 1] if g[0] > 0 else None
        pause_before = round((ws[0]["s"] - prv["e"]) if prv else 9.0, 3)
        rom = hinglish.romanize(text)
        sents.append({"id": len(sents), "s": ws[0]["s"], "e": ws[-1]["e"], "w0": g[0], "w1": g[-1] + 1,
                      "text": text, "roman": rom, "pause_before": pause_before, "pause_after": pause_after,
                      "punct_end": ws[-1]["w"].endswith(end_p)})
    for s in sents:
        s["complete"] = bool(lex.sentence_complete(s))
    return sents


# ---------------------------------------------------------------- diarization (lightweight)
def diarize(x: np.ndarray, sents: list[dict], f0: np.ndarray, f0_hop: float, cfg_d: dict) -> tuple[list[int], dict]:
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    feats, idx = [], []
    for s in sents:
        if s["e"] - s["s"] < cfg_d["min_sentence_s"]:
            continue
        y = x[int(s["s"] * SR):int(s["e"] * SR)]
        m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512, hop_length=160)
        d = librosa.feature.delta(m) if m.shape[1] >= 9 else np.zeros_like(m)
        f = f0[int(s["s"] / f0_hop):int(s["e"] / f0_hop)]
        f = f[~np.isnan(f)]
        lf = np.log(np.median(f)) if len(f) > 5 else np.nan
        feats.append(np.concatenate([m[1:].mean(1), m[1:].std(1), np.abs(d[1:]).mean(1), [lf]]))
        idx.append(s["id"])
    labels = [-1] * len(sents)
    if len(feats) < 2:
        return [0] * len(sents), {"speakers": 1}
    F = np.array(feats)
    col_med = np.nanmedian(F, 0)
    F = np.where(np.isnan(F), col_med, F)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    F[:, -1] *= 3.0   # pitch is the most speaker-discriminative cheap cue (male / female / child)
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=cfg_d["distance_threshold"] * np.sqrt(F.shape[1] / 20),
                                 linkage="ward").fit(F)
    lab = cl.labels_
    if lab.max() + 1 > cfg_d["max_speakers"]:
        lab = AgglomerativeClustering(n_clusters=cfg_d["max_speakers"], linkage="ward").fit(F).labels_
    # canonical label order = order of first appearance (determinism of ids)
    remap, nxt = {}, 0
    for l in lab:
        if l not in remap:
            remap[l] = nxt; nxt += 1
    for i, l in zip(idx, lab):
        labels[i] = remap[l]
    # short sentences inherit the previous speaker
    last = 0
    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = last
        last = labels[i]
    return labels, {"speakers": nxt, "clustered_sentences": len(idx)}
