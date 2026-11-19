"""9:16 canvas matching the reference screenshot: white header with a bold black title + emojis, footage
full-width in the middle, white footer band, an emoji sticker near the top of the footage, small captions."""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")


@dataclass
class Geometry:
    W: int
    H: int
    header_h: int
    video_y: int
    video_h: int

    @property
    def video_w(self):
        return self.W

    @property
    def aspect(self):
        return self.W / self.video_h


def geometry(W: int, H: int, cfg: dict) -> Geometry:
    hh = int(round(H * cfg["header_frac"]))
    vh = int(round(H * cfg["video_frac"]))
    vh -= vh % 2
    return Geometry(W, H, hh, hh, vh)


def _font(path, size):
    return ImageFont.truetype(path, size)


def _split(text: str):
    """[(segment, is_emoji)] so text and emoji use different fonts."""
    out, pos = [], 0
    for m in EMOJI_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _measure(segs, ft, fe):
    return sum((fe if e else ft).getlength(s) for s, e in segs)


def _draw_line(draw, img, x, y, segs, ft, fe, fill, emoji_px):
    for s, e in segs:
        if e:
            for ch in [c for c in s if c not in "️‍"]:
                em = render_emoji(ch, emoji_px, fe.path)
                img.paste(em, (int(x), int(y + ft.size * 0.62 - em.size[1] / 2)), em)
                x += emoji_px * 1.05
        else:
            draw.text((x, y), s, font=ft, fill=fill)
            x += ft.getlength(s)


_EMO_CACHE: dict = {}


def render_emoji(ch: str, px: int, font_path: str) -> Image.Image:
    key = (ch, px)
    if key not in _EMO_CACHE:
        f = ImageFont.truetype(font_path, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        bb = im.getbbox() or (0, 0, 160, 160)
        im = im.crop(bb)
        s = px / max(im.size)
        _EMO_CACHE[key] = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS)
    return _EMO_CACHE[key]


def base_canvas(g: Geometry, title: str, cfg: dict) -> np.ndarray:
    """White canvas with the header title drawn (BGR uint8). Footage is pasted per frame."""
    img = Image.new("RGB", (g.W, g.H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fe = _font(cfg["emoji_font"], 40)
    segs = _split(title)
    max_w = g.W * (1 - 2 * cfg["title_margin_frac"])
    size = int(g.H * cfg["title_size_frac"])
    lines = [segs]
    while size > g.H * 0.016:
        ft = _font(cfg["title_font"], size)
        if _measure(segs, ft, fe) <= max_w:
            break
        size -= 2
    ft = _font(cfg["title_font"], size)
    if _measure(segs, ft, fe) > max_w:            # two lines
        words = title.split(" ")
        best = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wmax = max(_measure(_split(a), ft, fe), _measure(_split(b), ft, fe))
            if best is None or wmax < best[0]:
                best = (wmax, a, b)
        size = int(g.H * cfg["title_size_frac"] * 0.85)
        ft = _font(cfg["title_font"], size)
        while max(_measure(_split(best[1]), ft, fe), _measure(_split(best[2]), ft, fe)) > max_w and size > 20:
            size -= 2; ft = _font(cfg["title_font"], size)
        lines = [_split(best[1]), _split(best[2])]
    emoji_px = int(size * 1.15)
    line_h = int(size * 1.25)
    y = (g.header_h - line_h * len(lines)) / 2 + size * 0.05
    for segs_l in lines:
        wl = sum((emoji_px * 1.05 * len([c for c in s if c not in "️‍"])) if e else ft.getlength(s) for s, e in segs_l)
        _draw_line(d, img, (g.W - wl) / 2, y, segs_l, ft, fe, (0, 0, 0), emoji_px)
        y += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class Overlay:
    """Pre-rendered RGBA sticker + caption images, alpha-composited onto the footage region."""

    def __init__(self, g: Geometry, cfg: dict, sticker: str | None):
        self.g, self.cfg = g, cfg
        self.sticker = None
        if sticker:
            px = int(g.W * cfg["sticker_size_frac"])
            self.sticker = np.array(render_emoji(sticker, px, cfg["emoji_font"]).convert("RGBA"))
        self.cap_font = _font(cfg["caption_font"], int(g.H * cfg["caption_size_frac"]))
        self._cap_cache: dict = {}
        import threading
        self._lock = threading.Lock()

    def caption_img(self, words: list[str], hi: int) -> np.ndarray:
        key = (tuple(words), hi)
        with self._lock:
            if key in self._cap_cache:
                return self._cap_cache[key]
        f = self.cap_font
        stroke = max(2, int(f.size * 0.09))
        sp = f.getlength(" ")
        widths = [f.getlength(w) for w in words]
        W = int(sum(widths) + sp * (len(words) - 1) + 4 * stroke)
        H = int(f.size * 1.4 + 2 * stroke)
        im = Image.new("RGBA", (max(1, W), H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x = 2 * stroke
        for k, (w, ww) in enumerate(zip(words, widths)):
            col = tuple(self.cfg["caption_highlight"]) if k == hi else (255, 255, 255)
            d.text((x, stroke), w, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += ww + sp
        arr = np.array(im)
        with self._lock:
            if len(self._cap_cache) > 400:
                self._cap_cache.clear()
            self._cap_cache[key] = arr
        return arr

    @staticmethod
    def blend(dst: np.ndarray, rgba: np.ndarray, x: int, y: int, scale: float = 1.0):
        if scale != 1.0:
            h, w = rgba.shape[:2]
            rgba = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = rgba.shape[:2]
        H, W = dst.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        a = src[..., 3:4].astype(np.float32) / 255
        roi = dst[y0:y1, x0:x1].astype(np.float32)
        rgb = src[..., 2::-1].astype(np.float32)   # RGBA -> BGR
        dst[y0:y1, x0:x1] = (roi * (1 - a) + rgb * a + 0.5).astype(np.uint8)


def caption_groups(words: list[dict], t0: float, t1: float, max_words: int, max_gap: float) -> list[dict]:
    """2-4 word groups; each word keeps its own timing for the highlight."""
    ws = [w for w in words if t0 <= w["s"] < t1]
    groups, cur = [], []
    for w in ws:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > max_gap):
            groups.append(cur); cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    out = []
    for k, g in enumerate(groups):
        end = min(g[-1]["e"] + 0.4, groups[k + 1][0]["s"] if k + 1 < len(groups) else g[-1]["e"] + 0.4)
        out.append({"s": g[0]["s"], "e": end, "words": g})
    return out
