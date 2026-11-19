"""Micro jump cuts ONLY inside fully silent stretches.

A stretch qualifies when Silero VAD says no speech for >= min_silence_s, no word is inside it, and there is no
laugh / scream / cheer in it. We keep `keep_pad_s` of silence at each side and cut the middle, so speech and
reactions are never touched and lip-sync cannot drift (every kept segment is frame-exact, audio sample-exact).
"""
from __future__ import annotations

import numpy as np

from . import cut


def silent_gaps(A: dict, t0: float, t1: float, c: dict) -> list[tuple[float, float]]:
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
            if b - a >= c["min_silence_s"]:
                gaps.append((a, b))
            i = j
        else:
            i += 1
    out = []
    words = A["words"]
    for a, b in gaps:
        if any(a < w["e"] and w["s"] < b for w in words):
            continue
        s0, s1 = int(a), int(np.ceil(b))
        if max(ps["laugh"][s0:s1].max(initial=0), ps["scream"][s0:s1].max(initial=0), ps["cheer"][s0:s1].max(initial=0)) >= 0.3:
            continue
        out.append((a, b))
    return out


def plan(A: dict, start_abs: float, end_abs: float, off: float, fps: int, c: dict) -> tuple[list[cut.Span], list[dict]]:
    """Keep-segments (clip-time Spans, frame-snapped) + the list of cuts made (absolute times)."""
    full = cut.snap(start_abs - off, end_abs - off, fps)
    if not c.get("enabled"):
        return [full], []
    pad, guard = c["keep_pad_s"], c["edge_guard_s"]
    cuts = []
    for a, b in silent_gaps(A, start_abs + guard, end_abs - guard, c):
        ca, cb = a + pad, b - pad
        if cb - ca >= 0.3:
            cuts.append((ca, cb))
    budget = c["max_cut_frac"] * (end_abs - start_abs)
    cuts.sort(key=lambda x: -(x[1] - x[0]))                 # longest silences first, within the budget
    kept, used = [], 0.0
    for a, b in cuts:
        if used + (b - a) <= budget:
            kept.append((a, b)); used += b - a
    kept.sort()
    segs, cur = [], full.fa
    log = []
    for a, b in kept:
        fa, fb = int(round((a - off) * fps)), int(round((b - off) * fps))
        if fa - cur >= 6:
            segs.append(cut.Span(cur, fa, fps)); cur = fb
            log.append({"cut_from": round(a, 3), "cut_to": round(b, 3), "removed_s": round((fb - fa) / fps, 3)})
    segs.append(cut.Span(cur, full.fb, fps))
    return [s for s in segs if s.fb > s.fa], log
