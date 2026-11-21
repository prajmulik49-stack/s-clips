"""Micro jump cuts ONLY where nobody talks.

silence: Silero VAD says no speech for >= min_silence_s, no word is inside the part we cut, and there is no
  laugh / scream / cheer in it. We keep `keep_pad_s` of silence at each side and cut the middle.
music:   no speech for >= music_min_s while YAMNet hears music (only music playing): a small middle part
  (music_cut_frac, keeping >= music_keep_s each side) is cut, so a music bit gets shorter, not removed.
Speech and reactions are never touched and lip-sync cannot drift (every kept segment is frame-exact, audio
sample-exact). YouTube caption words carry padded timings (a line's last word is stretched to 1.2 s), so a word
only blocks a cut over its first `word_max_s`.
"""
from __future__ import annotations

import numpy as np

from . import cut


def _word_inside(words, a: float, b: float, max_w: float) -> bool:
    return any(not w.get("_tag") and w["s"] < b and a < min(w["e"], w["s"] + max_w) for w in words)


def _mark_tags(words: list[dict]) -> None:
    """YouTube caption sound tags ("[संगीत]" = music, "[चीखने की आवाज़]" = screaming) are not speech: mark their
    words `_tag` ("music" for संगीत / music, else "sound")."""
    i = 0
    while i < len(words):
        if words[i]["w"].startswith("["):
            j = i
            while j < len(words) - 1 and not words[j]["w"].endswith("]") and j - i < 6:
                j += 1
            txt = " ".join(w["w"] for w in words[i:j + 1])
            kind = "music" if ("संगीत" in txt or "music" in txt.lower()) else "sound"
            for w in words[i:j + 1]:
                w["_tag"] = kind
            i = j + 1
        else:
            i += 1


def silent_gaps(A: dict, t0: float, t1: float, c: dict) -> list[tuple[float, float, str]]:
    """(start, end, kind) of no-speech stretches inside [t0, t1]; kind = "silence" | "music"."""
    vad = A["vad_prob"]
    hop = 0.032
    ps = A["per_sec"]
    speech = vad >= 0.5
    gaps = []
    i0, i1 = int(t0 / hop), int(t1 / hop)
    i = i0
    while i < i1:
        if not speech[i]:
            j = i
            while j < i1 and not speech[j]:
                j += 1
            a, b = i * hop, j * hop
            if b - a >= min(c["min_silence_s"], c.get("music_min_s", 99)):
                gaps.append((a, b))
            i = j
        else:
            i += 1
    out = []
    words = A["words"]
    from .analysis.topics import robust_z
    if "_jc_z" not in A:   # shouts / laughs are not "speech" for the VAD but must never be cut
        A["_jc_z"] = np.maximum(np.maximum(robust_z(ps["scream"]), robust_z(ps["laugh"])), robust_z(ps["cheer"]))
    rms10 = A.get("rms10")
    quiet_thr = None
    if rms10 is not None:
        seg = rms10[int(t0 * 100):int(t1 * 100)]
        quiet_thr = float(np.percentile(seg, c.get("quiet_percentile", 45))) if len(seg) else None
    pad, max_w = c["keep_pad_s"], c.get("word_max_s", 0.6)
    if "_jc_tags" not in A:
        _mark_tags(words); A["_jc_tags"] = True
    words = [w for w in words if w["s"] < t1 + 2 and w["e"] > t0 - 2]
    music = ps.get("music")
    for a, b in gaps:
        s0, s1 = int(a), int(np.ceil(b))
        if A["_jc_z"][s0:s1].max(initial=0) >= 2.0:
            continue
        mus = float(np.mean(music[s0:s1])) if music is not None and s1 > s0 else 0.0
        if any(w.get("_tag") == "music" and w["s"] < b and a < w["e"] for w in words):
            mus = max(mus, 1.0)                     # the captions themselves say [संगीत]
        if (c.get("music_enabled", True) and b - a >= c.get("music_min_s", 2.0) and mus >= c.get("music_min_prob", 0.5)
                and not _word_inside(words, a + c.get("music_keep_s", 1.0), b - c.get("music_keep_s", 1.0), max_w)):
            out.append((a, b, "music"))
            continue
        if b - a < c["min_silence_s"] or _word_inside(words, a + pad, b - pad, max_w):
            continue
        if quiet_thr is not None:
            r = rms10[int(a * 100):int(b * 100)]
            if len(r) and (r <= quiet_thr + 3.0).mean() < 0.8:      # not actually quiet (shouting, crash)
                continue
        out.append((a, b, "silence"))
    return out


def plan(A: dict, start_abs: float, end_abs: float, off: float, fps: int, c: dict) -> tuple[list[cut.Span], list[dict]]:
    """Keep-segments (clip-time Spans, frame-snapped) + the list of cuts made (absolute times)."""
    full = cut.snap(start_abs - off, end_abs - off, fps)
    if not c.get("enabled"):
        return [full], []
    pad, guard = c["keep_pad_s"], c["edge_guard_s"]
    cuts = []
    for a, b, kind in silent_gaps(A, start_abs + guard, end_abs - guard, c):
        if kind == "music":
            L = b - a
            rm = min(c.get("music_cut_frac", 0.4) * L, L - 2 * c.get("music_keep_s", 1.0))
            ca = a + (L - rm) / 2
            cb = ca + rm
        else:
            ca, cb = a + pad, b - pad
        if cb - ca >= 0.3:
            cuts.append((ca, cb, kind))
    dur = end_abs - start_abs
    budget = max(0.0, min(c["max_cut_frac"] * dur, dur - c.get("min_out_s", 0)))    # never below min_out_s
    cuts.sort(key=lambda x: -(x[1] - x[0]))                 # longest pauses first, within the budget
    kept, used = [], 0.0
    for a, b, kind in cuts:
        if used + (b - a) <= budget:
            kept.append((a, b, kind)); used += b - a
    kept.sort()
    segs, cur = [], full.fa
    log = []
    for a, b, kind in kept:
        fa, fb = int(round((a - off) * fps)), int(round((b - off) * fps))
        if fa - cur >= 6:
            segs.append(cut.Span(cur, fa, fps)); cur = fb
            log.append({"cut_from": round(a, 3), "cut_to": round(b, 3), "removed_s": round((fb - fa) / fps, 3), "kind": kind})
    segs.append(cut.Span(cur, full.fb, fps))
    return [s for s in segs if s.fb > s.fa], log
