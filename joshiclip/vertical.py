"""Render one vertical Short: decode the exact span from norm.mp4 (one trim graph) -> face-follow crop -> grade + glow
-> canvas (header title, sticker, captions) -> x264, muxed with the processed voice. Frame count asserted."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import camera, cut, layout, look, media, tools, voice
from .selection import moment_type
from .text import hinglish

STICKER = {"funny": "😂", "danger": "😱", "emotional": "🥹", "reveal": "😳", "question": "🤔", "conflict": "😤",
           "cheer": "🥳", "name_call": "😳"}


def _decoder(norm: Path, span: cut.Span, w: int, h: int):
    g = cut.trim_graph(span, vout="vt").split(";")[0]
    g += f";[vt]scale={w}:{h}:flags=lanczos,format=bgr24[v]"
    p = subprocess.Popen([media.ff(), "-v", "error", *cut.seek_args(span), "-i", str(norm), "-filter_complex", g,
                          "-map", "[v]", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, env=tools.child_env(),
                         bufsize=w * h * 3 * 4)
    return p


def _read(p, n):
    buf = bytearray()
    while len(buf) < n:
        b = p.stdout.read(n - len(buf))
        if not b:
            return None
        buf += b
    return bytes(buf)


def _sample_frames(norm: Path, span: cut.Span, k: int, w: int, h: int) -> list[np.ndarray]:
    out = []
    for i in range(k):
        t = span.start + (i + 0.5) * span.dur / k
        r = media.run([media.ff(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(norm), "-frames:v", "1", "-vf",
                       f"scale={w}:{h}:flags=area,format=bgr24", "-f", "rawvideo", "-"])
        out.append(np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3))
    return out


def render(A: dict, c: dict, out: Path, cfg, mode: str, log=print) -> dict:
    L = cfg["layout"]
    q = cfg["render"][mode]
    W, H = (L["canvas_final"] if mode == "final" else L["canvas_draft"])
    g = layout.geometry(W, H, L)
    fps = A["fps"]
    span = cut.snap(c["start"], c["end"], fps)
    n = span.fb - span.fa
    norm = A["workdir"] / "norm.mp4"
    tmp = out.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    rep = {"canvas": [W, H], "span": [span.start, span.end], "frames_expected": n}
    T = {}

    # ---- audio
    t = time.perf_counter()
    src_wav, voc_wav = tmp / "src.wav", tmp / "voice.wav"
    voice.extract_span(norm, span, src_wav)
    rep["audio"] = voice.process(src_wav, voc_wav, cfg["voice"], log)
    T["voice"] = round(time.perf_counter() - t, 1)

    # ---- camera
    src_h = A["meta"]["height"]; src_w = A["meta"]["width"]
    dec_h = min(src_h, int(np.ceil(g.video_h * 1.25 / 2) * 2))     # decode a bit above output res for the crop
    dec_w = int(round(dec_h * src_w / src_h / 2) * 2)
    cam = camera.path(A["vision"], span.start, n, fps, src_w / src_h, g.aspect, A["scene_cuts"], c["peak_t"], cfg["camera"])
    rep["camera"] = {"face_frames": round(cam["face_frames"], 3), "zoom_max": round(float(cam["zoom"].max()), 4)}

    # ---- grade (tuned once per Short on sample frames of the actual crop)
    t = time.perf_counter()
    ref = json.loads(Path(cfg.path("style_ref_metrics")).read_text())["video_area"]
    samples = []
    for f in _sample_frames(norm, span, 8, dec_w, dec_h):
        cw = int(round(g.aspect * dec_h)); x0 = (dec_w - cw) // 2
        samples.append(cv2.resize(f[:, x0:x0 + cw], (480, int(480 / g.aspect)), interpolation=cv2.INTER_AREA))
    params, before, after, target = look.tune(samples, ref, look.GradeParams(**cfg["look"]), tonal_strength=cfg["look_tune"]["tonal_strength"])
    grader = look.Grader(params)
    rep["grade"] = {"params": {k: round(v, 4) for k, v in params.__dict__.items()},
                    "before": {k: round(v, 2) for k, v in before.items()}, "after": {k: round(v, 2) for k, v in after.items()},
                    "reference": ref, "target": {k: round(v, 2) for k, v in target.items()},
                    "within_10pct": look.within(after, target)}
    T["grade_tune"] = round(time.perf_counter() - t, 1)

    # ---- overlays
    canvas0 = layout.base_canvas(g, c["title"], L)
    sticker = STICKER.get(moment_type(c), "😂") if L["sticker"] else None
    ov = layout.Overlay(g, L, sticker)
    groups = layout.caption_groups(A["words"], span.start, span.end, L["caption_max_words"], 0.6) if L["captions"] else []
    vis = A["vision"]; vfps = float(vis["fps"])

    # ---- encode
    enc = subprocess.Popen([media.ff(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-i", str(voc_wav), "-map", "0:v", "-map", "1:a",
                            *cut.encode_args(q, fps, cfg["render"]["x264_threads"]), "-shortest", str(out)],
                           stdin=subprocess.PIPE, env=tools.child_env())
    dec = _decoder(norm, span, dec_w, dec_h)
    fb = dec_w * dec_h * 3
    t = time.perf_counter()
    written = 0
    cap_y_mid = g.video_y + int(g.video_h * L["caption_y_frac"])
    cap_y_low = g.video_y + int(g.video_h * L["caption_y_alt_frac"])
    import bisect
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    g_starts = [gr["s"] for gr in groups]
    cap_words = [[hinglish.romanize(w["w"]) for w in gr["words"]] for gr in groups]
    cv2.setNumThreads(2)

    def process(i: int, raw: bytes) -> bytes:
        fr = np.frombuffer(raw, np.uint8).reshape(dec_h, dec_w, 3)
        ch = dec_h / cam["zoom"][i]
        cw = ch * g.aspect
        cx = cam["cx"][i] * dec_w
        x0 = float(np.clip(cx - cw / 2, 0, dec_w - cw)); y0 = (dec_h - ch) / 2
        M = np.array([[cw / g.W, 0, x0], [0, ch / g.video_h, y0]], np.float64)
        crop = cv2.warpAffine(fr, M, (g.W, g.video_h), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        k = int(round((span.start + i / fps) * vfps))
        faces_px = []
        if 0 <= k < len(vis["faces"]):
            for fx, fy, fw, fh, _sc in vis["faces"][k]:
                if fw <= 0:
                    continue
                X0 = (fx * dec_w - x0) * g.W / cw; X1 = ((fx + fw) * dec_w - x0) * g.W / cw
                Y0 = (fy * dec_h - y0) * g.video_h / ch; Y1 = ((fy + fh) * dec_h - y0) * g.video_h / ch
                if X1 > 0 and X0 < g.W:
                    faces_px.append((X0, Y0, X1, Y1))
        crop = grader.apply(crop, faces_px)
        frame = canvas0.copy()
        frame[g.video_y:g.video_y + g.video_h] = crop
        tt = i / fps
        if ov.sticker is not None:
            sc = min(1.0, 0.4 + 0.6 * (tt / 0.25)) if tt < 0.25 else 1.0
            sh, sw = ov.sticker.shape[:2]
            ov.blend(frame, ov.sticker, int((W - sw * sc) / 2), int(g.video_y + g.video_h * L["sticker_y_frac"] - sh * sc / 2), sc)
        ts = span.start + tt
        gi = bisect.bisect_right(g_starts, ts) - 1
        if gi >= 0 and groups[gi]["s"] <= ts < groups[gi]["e"]:
            gr = groups[gi]
            hi = max([j for j, w in enumerate(gr["words"]) if w["s"] <= ts] or [0])
            img = ov.caption_img(cap_words[gi], hi)
            ih, iw = img.shape[:2]
            yb = cap_y_mid
            if any(fy0 < yb + ih / 2 + 10 and fy1 > yb - ih / 2 - 10 for _, fy0, _, fy1 in
                   [(f0[0], f0[1] + g.video_y, f0[2], f0[3] + g.video_y) for f0 in faces_px]):
                yb = cap_y_low
            ov.blend(frame, img, int((W - iw) / 2), int(yb - ih / 2))
        return frame.tobytes()

    workers = cfg["render"].get("frame_workers", 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q = deque()
        for i in range(n):
            raw = _read(dec, fb)
            if raw is None:
                break
            q.append(ex.submit(process, i, raw))
            while len(q) > 2 * workers:
                enc.stdin.write(q.popleft().result()); written += 1
        while q:
            enc.stdin.write(q.popleft().result()); written += 1
    dec.stdout.close(); dec.wait()
    enc.stdin.close(); enc.wait()
    T["frames"] = round(time.perf_counter() - t, 1)
    rep["frames_written"] = written
    rep["render_fps"] = round(written / max(1e-6, T["frames"]), 2)
    if written != n:
        raise RuntimeError(f"decoder delivered {written} frames, expected {n} - refusing to write a drifting Short")
    if enc.returncode != 0:
        raise RuntimeError("encoder failed")
    rep["timings"] = T
    rep["_vgeom"] = {"W": W, "H": H, "video_y": g.video_y, "video_h": g.video_h, "cx": cam["cx"].tolist(),
                     "zoom": cam["zoom"].tolist(), "frame0": span.fa}
    return rep
