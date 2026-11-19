"""HDR-style grade + highlight glow, auto-tuned per clip to the reference screenshot's metrics.

Grade (per frame, fixed parameters per Short -> no flicker):
  1. tone LUT on LAB L: black point, shadow lift, midtone gamma, highlight shoulder (compression)
  2. CLAHE local contrast on L, blended at a fixed strength
  3. vibrance on a/b with skin-tone protection
  4. light unsharp mask (after scaling)
Glow: blurred highlight layer screen-blended at ~12 %, suppressed inside face boxes (faces stay sharp).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


def metrics(bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) / 2.55
    blur = cv2.GaussianBlur(L, (0, 0), max(2.0, bgr.shape[1] / 60))
    a = lab[..., 1].astype(np.float32) - 128
    b = lab[..., 2].astype(np.float32) - 128
    mid = (L > 20) & (L < 85)                      # perceptual colourfulness: mid-tones only (dark noise is not colour)
    chroma = np.sqrt(a * a + b * b)
    return {"L_p5": float(np.percentile(L, 5)), "L_p50": float(np.percentile(L, 50)), "L_p95": float(np.percentile(L, 95)),
            "local_contrast": float(np.abs(L - blur).mean()), "chroma_mid": float(chroma[mid].mean()) if mid.any() else 0.0}


@dataclass
class GradeParams:
    black: float = 0.0        # L units (0-100) mapped to 0
    white: float = 100.0      # L units mapped to the shoulder top
    gamma: float = 1.0        # midtone gamma (<1 brightens)
    shoulder: float = 0.35    # highlight compression strength
    lift: float = 0.03        # shadow lift (fraction of range)
    clahe: float = 1.6        # CLAHE clip limit
    clahe_mix: float = 0.6
    vibrance: float = 1.25
    sharpen: float = 0.35
    glow: float = 0.12
    glow_thr: float = 0.75    # L fraction where the glow starts


def _lut(p: GradeParams) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.clip((x * 100 - p.black) / max(1e-3, (p.white - p.black)), 0, 1)
    y = y ** p.gamma
    y = p.lift + (1 - p.lift) * y
    # soft shoulder: compress highlights above 0.7
    k = 0.7
    hi = y > k
    y[hi] = k + (1 - k) * (1 - np.exp(-(y[hi] - k) / (1 - k) * (1 + 3 * p.shoulder))) / (1 - np.exp(-(1 + 3 * p.shoulder)))
    # gentle S for punch
    y = y + 0.08 * np.sin((y - 0.5) * np.pi) * (1 - p.lift)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


class Grader:
    def __init__(self, p: GradeParams):
        self.p = p
        self.lut = _lut(p)
        self.clahe = cv2.createCLAHE(clipLimit=p.clahe, tileGridSize=(8, 8))

    def apply(self, bgr: np.ndarray, faces_px: list | None = None) -> np.ndarray:
        """uint8 / quarter-resolution math throughout (~25 ms at 1080x1580)."""
        p = self.p
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        Lc, ac, bc = cv2.split(lab)
        L = cv2.LUT(Lc, self.lut)
        if p.clahe_mix > 0:
            L = cv2.addWeighted(self.clahe.apply(L), p.clahe_mix, L, 1 - p.clahe_mix, 0)
        # vibrance gain map at 1/4 res: more boost for dull colours, less on skin
        qs = (max(1, w // 4), max(1, h // 4))
        aq = cv2.resize(ac, qs, interpolation=cv2.INTER_AREA).astype(np.float32) - 128
        bq = cv2.resize(bc, qs, interpolation=cv2.INTER_AREA).astype(np.float32) - 128
        chroma = cv2.magnitude(aq, bq)
        skin = ((aq > 4) & (bq > 0.5 * aq) & (bq < 3.2 * aq + 6) & (chroma > 6) & (chroma < 45)).astype(np.float32)
        skin = cv2.GaussianBlur(skin, (0, 0), 2)
        gain = 1 + (p.vibrance - 1) * np.clip(1 - chroma / 70, 0, 1) * (1 - 0.7 * skin)
        gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
        a2 = np.clip(cv2.multiply(ac.astype(np.float32) - 128, gain) + 128, 0, 255).astype(np.uint8)
        b2 = np.clip(cv2.multiply(bc.astype(np.float32) - 128, gain) + 128, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([L, a2, b2]), cv2.COLOR_LAB2BGR)
        if p.glow > 0:
            out = self._glow(out, L, faces_px)
        if p.sharpen > 0:
            bl = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1 + p.sharpen, bl, -p.sharpen, 0)
        return out

    def _glow(self, img, L, faces_px):
        """Screen-blend a blurred highlight layer: out = img + glow * layer * (255 - img) / 255 (uint8 ops)."""
        p = self.p
        h, w = L.shape
        qs = (max(1, w // 4), max(1, h // 4))
        sm = cv2.resize(L, qs, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        m = np.clip((sm - p.glow_thr) / (1 - p.glow_thr), 0, 1)
        if faces_px:
            fm = np.ones_like(m)
            for x0, y0, x1, y1 in faces_px:
                cv2.rectangle(fm, (int(x0 / 4), int(y0 / 4)), (int(x1 / 4), int(y1 / 4)), 0.15, -1)
            m *= cv2.GaussianBlur(fm, (0, 0), 4)
        small = cv2.resize(img, qs, interpolation=cv2.INTER_AREA).astype(np.float32)
        layer = cv2.GaussianBlur(small * m[..., None], (0, 0), max(4, w / 120))
        layer = cv2.resize(np.clip(layer, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        add = cv2.multiply(layer, cv2.bitwise_not(img), scale=p.glow / 255.0)
        return cv2.add(img, add)


def within(m: dict, ref: dict, tol: float = 0.10) -> dict:
    """Per-metric pass: +-10 % of the reference (absolute floor of 3 L units for the near-zero black point)."""
    out = {}
    for k, r in ref.items():
        if k not in m:
            continue
        lim = max(abs(r) * tol, 3.0 if k.startswith("L_") else 0.5)
        out[k] = abs(m[k] - r) <= lim
    return out


def tune(frames: list[np.ndarray], ref: dict, start: GradeParams | None = None, iters: int = 14,
         tonal_strength: float = 0.65) -> tuple[GradeParams, dict, dict, dict]:
    """Deterministic proportional search. Colour (chroma_mid) and local contrast aim at the reference; tonal
    percentiles move `tonal_strength` of the way there, because forcing another scene's histogram onto a bright
    room crushes faces. Returns (params, before, after, target)."""
    p = GradeParams(**asdict(start or GradeParams()))
    glow, sharpen = p.glow, p.sharpen
    p.glow = 0.0; p.sharpen = 0.0

    def meas(pp):
        gr = Grader(pp)
        ms = [metrics(gr.apply(f)) for f in frames]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    before = {k: float(np.mean([metrics(f)[k] for f in frames])) for k in ref}
    target = {k: (before[k] + tonal_strength * (ref[k] - before[k])) if k.startswith("L_") else ref[k] for k in ref}
    m = meas(p)
    for _ in range(iters):
        if all(within(m, target).values()):
            break
        p.black = float(np.clip(p.black + 0.6 * (m["L_p5"] - target["L_p5"]), -5, 10))
        p.white = float(np.clip(p.white + 0.8 * (m["L_p95"] - target["L_p95"]), 85, 110))
        p.gamma = float(np.clip(p.gamma * (m["L_p50"] / max(1.0, target["L_p50"])) ** 0.8, 0.85, 1.1))
        r = target["local_contrast"] / max(0.5, m["local_contrast"])
        p.clahe = float(np.clip(p.clahe * r ** 1.2, 1.0, 2.6))
        p.clahe_mix = float(np.clip(p.clahe_mix * r ** 0.5, 0.4, 0.75))
        p.vibrance = float(np.clip(p.vibrance * (target["chroma_mid"] / max(1.0, m["chroma_mid"])), 1.0, 1.8))
        m = meas(p)
    p.glow, p.sharpen = glow, sharpen
    return p, before, m, target
