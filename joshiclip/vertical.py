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


def segments_for(A: dict, c: dict, cfg, off: float, fps: int):
    from . import jumpcuts
    return jumpcuts.plan(A, c["start"], c["end"], off, fps, cfg["jumpcuts"])


def seg_map(segs) -> list[list[float]]:
    """[(out_t0, src_t0, dur, out_f0, src_f0)] for each kept segment (sync checks probe inside segments)."""
    out, f = [], 0
    for s_ in segs:
        out.append([f / s_.fps, s_.start, s_.dur, f, s_.fa]); f += s_.fb - s_.fa
    return out


def render(A: dict, c: dict, out: Path, cfg, mode: str, log=print, src: dict | None = None,
           voice_wav: Path | None = None, voice_rep: dict | None = None) -> dict:
    """src (fast mode): {"path": 4K clip, "off": analysis_time - clip_time, "fps", "width", "height"}."""
    L = cfg["layout"]
    q = cfg["render"][mode]
    W, H = (L["canvas_final"] if mode == "final" else L["canvas_draft"])
    g = layout.geometry(W, H, L)
    off = src["off"] if src else 0.0
    fps = src["fps"] if src else A["fps"]
    span = cut.snap(c["start"] - off, c["end"] - off, fps)
    segs, jc_log = segments_for(A, c, cfg, off, fps)
    idx = np.concatenate([np.arange(s_.fa, s_.fb) for s_ in segs])   # source (clip) frame of every output frame
    n = len(idx)
    times = idx / fps + off                         # analysis-timeline time of every output frame
    norm = Path(src["path"]) if src else A["workdir"] / "norm.mp4"
    t_abs0 = span.start + off
    tmp = out.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    rep = {"canvas": [W, H], "span": [span.start, span.end], "frames_expected": n,
           "jump_cuts": jc_log, "segments": [[s_.start, s_.end] for s_ in segs]}
    T = {}

    # ---- audio
    t = time.perf_counter()
    if voice_wav is not None:
        voc_wav = voice_wav
        rep["audio"] = voice_rep
    else:
        src_wav, voc_wav = tmp / "src.wav", tmp / "voice.wav"
        voice.extract_segments(norm, segs, src_wav, cfg["jumpcuts"].get("fade_ms", 0))
        rep["audio"] = voice.process(src_wav, voc_wav, dict(cfg["voice"], music=cfg.get("music"),
                                                            music_seed=f"{A['vid']}:{c['start']:.3f}"), log)
    T["voice"] = round(time.perf_counter() - t, 1)

    # ---- camera
    src_h = src["height"] if src else A["meta"]["height"]
    src_w = src["width"] if src else A["meta"]["width"]
    cam = camera.path(A["vision"], t_abs0, n, fps, src_w / src_h, g.aspect, A["scene_cuts"], c["peak_t"], cfg["camera"],
                      times=times)
    rep["camera"] = {"face_frames": round(cam["face_frames"], 3), "zoom_max": round(float(cam["zoom"].max()), 4)}
    # ffmpeg does the moving crop (zoom-1 strip, full source height); Python only applies the subtle zoom
    CW = int(round(src_h * g.aspect / 2) * 2); CH = src_h
    strip_x = np.clip(np.round(cam["cx"] * src_w - CW / 2), 0, src_w - CW).astype(int)
    pivot = np.clip((cam["cx"] * src_w - strip_x) / CW, 0.0, 1.0)          # zoom centre inside the strip (0..1)

    # ---- grade: tune on sample frames of the actual crop, bake the per-pixel part into a 3D LUT
    t = time.perf_counter()
    ref = json.loads(Path(cfg.path("style_ref_metrics")).read_text())["video_area"]
    sh_ = 702; sw_ = int(round(src_w * sh_ / src_h / 2) * 2); cws = int(round(sh_ * g.aspect))
    samples = []
    ns = cfg["render"].get("grade_samples", 5)
    for j, f in enumerate(_sample_frames(norm, span, ns, sw_, sh_)):
        i = int((j + 0.5) * n / ns)
        x0 = int(np.clip(cam["cx"][i] * sw_ - cws / 2, 0, sw_ - cws))
        samples.append(np.ascontiguousarray(f[:, x0:x0 + cws]))
    params, before, after, target = look.tune_fast(samples, ref, look.GradeParams(**cfg["look"]),
                                                   tonal_strength=cfg["look_tune"]["tonal_strength"],
                                                   color_strength=cfg["look_tune"].get("color_strength", 1.0))
    look.write_cube(params, tmp / "grade.cube")
    glow = look.Grader(params)
    rep["grade"] = {"params": {k: round(v, 4) for k, v in params.__dict__.items()},
                    "before": {k: round(v, 2) for k, v in before.items()}, "after_model": {k: round(v, 2) for k, v in after.items()},
                    "reference": ref, "target": {k: round(v, 2) for k, v in target.items()}}
    T["grade_tune"] = round(time.perf_counter() - t, 1)

    # ---- overlays
    canvas0 = layout.base_canvas(g, c["title"], L)
    sticker = STICKER.get(moment_type(c), "😂") if L["sticker"] else None
    ov = layout.Overlay(g, L, sticker)
    groups = layout.caption_groups(A["words"], t_abs0, t_abs0 + span.dur, L["caption_max_words"], 0.6) if L["captions"] else []
    vis = A["vision"]; vfps = float(vis["fps"])

    if not L["captions"]:
        T["mode"] = "single-ffmpeg-graph"
        return _render_ffmpeg(rep, T, cfg, q, g, W, H, fps, n, segs, norm, tmp, out, voc_wav, cam, CW, CH, src_w, src_h,
                              params, canvas0, ov, span, target, log)

    # ---- decoder: exact trim -> per-frame crop (sendcmd) -> scale -> 3D LUT -> sharpen (clarity + glow in Python)
    with open(tmp / "cam.cmd", "w", encoding="ascii", newline="\n") as f:
        for i in range(n):
            f.write(f"{max(0.0, (i - 0.5) / fps):.6f} crop@cam x {strip_x[i]};\n")
    vg = cut.multi_trim_graph(segs, vout="vt")
    vg += (f";[vt]sendcmd=f=cam.cmd,crop@cam=w={CW}:h={CH}:x={strip_x[0]}:y=0,"
           f"scale={g.W}:{g.video_h}:flags=lanczos,unsharp=5:5:{params.sharpen:.4f}:3:3:0,"
           f"format=gbrp,lut3d=file=grade.cube:interp=tetrahedral,format=bgr24[v]")
    dec = subprocess.Popen([media.ff(), "-v", "error", *cut.seek_args(segs[0]), "-i", str(Path(norm).resolve()),
                            "-filter_complex_threads", str(cfg["render"].get("decoder_filter_threads", 4)),
                            "-filter_complex", vg, "-map", "[v]", "-f", "rawvideo", "-"], stdout=subprocess.PIPE,
                           env=tools.child_env(), cwd=str(tmp), bufsize=g.W * g.video_h * 3 * 4)
    enc = subprocess.Popen([media.ff(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-i", str(voc_wav), "-map", "0:v", "-map", "1:a",
                            *cut.encode_args(q, fps, cfg["render"]["x264_threads"]), "-shortest", str(out)],
                           stdin=subprocess.PIPE, env=tools.child_env())
    fb = g.W * g.video_h * 3
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
    VW, VH = g.W, g.video_h

    def process(i: int, raw: bytes) -> bytes:
        crop = np.frombuffer(raw, np.uint8).reshape(VH, VW, 3)
        z = float(cam["zoom"][i]); cxp = float(pivot[i]) * VW; cyp = VH / 2
        if z > 1.002:   # zoom about the face pivot: out = p + (in - p) * z  (region resize, cheaper than a warp)
            rw, rh = VW / z, VH / z
            rx = min(max(cxp - cxp / z, 0.0), VW - rw); ry = min(max(cyp - cyp / z, 0.0), VH - rh)
            x0i, y0i = int(rx), int(ry)
            crop = cv2.resize(crop[y0i:y0i + int(np.ceil(rh)) + 1, x0i:x0i + int(np.ceil(rw)) + 1], (VW, VH),
                              interpolation=cv2.INTER_LINEAR)
            cxp = (cxp - x0i) * z; cyp = (cyp - y0i) * z        # pivot in output coords for the face mapping below
            cxp_src, cyp_src = x0i, y0i
        else:
            cxp_src = cyp_src = 0
        k = int(round(times[i] * vfps))
        faces_px = []
        if 0 <= k < len(vis["faces"]):
            for fx, fy, fw, fh, _sc in vis["faces"][k]:
                if fw <= 0:
                    continue
                X0 = (fx * src_w - strip_x[i]) * VW / CW; X1 = ((fx + fw) * src_w - strip_x[i]) * VW / CW
                Y0 = fy * VH; Y1 = (fy + fh) * VH
                if z > 1.002:
                    X0, X1 = (X0 - cxp_src) * z, (X1 - cxp_src) * z
                    Y0, Y1 = (Y0 - cyp_src) * z, (Y1 - cyp_src) * z
                if X1 > 0 and X0 < VW:
                    faces_px.append((X0, Y0, X1, Y1))
        crop = look.finish(np.ascontiguousarray(crop), params, faces_px)
        frame = canvas0.copy()
        frame[g.video_y:g.video_y + VH] = crop
        tt = i / fps
        if ov.sticker is not None:
            sc = min(1.0, 0.4 + 0.6 * (tt / 0.25)) if tt < 0.25 else 1.0
            sh, sw = ov.sticker.shape[:2]
            ov.blend(frame, ov.sticker, int((W - sw * sc) / 2), int(g.video_y + VH * L["sticker_y_frac"] - sh * sc / 2), sc)
        ts = times[i]
        gi = bisect.bisect_right(g_starts, ts) - 1
        if gi >= 0 and groups[gi]["s"] <= ts < groups[gi]["e"]:
            gr = groups[gi]
            hi = max([j for j, w in enumerate(gr["words"]) if w["s"] <= ts] or [0])
            img = ov.caption_img(cap_words[gi], hi)
            ih, iw = img.shape[:2]
            yb = cap_y_mid
            if any(fy0 + g.video_y < yb + ih / 2 + 10 and fy1 + g.video_y > yb - ih / 2 - 10 for _, fy0, _, fy1 in faces_px):
                yb = cap_y_low
            ov.blend(frame, img, int((W - iw) / 2), int(yb - ih / 2))
        return frame.tobytes()

    workers = cfg["render"].get("frame_workers", 4)
    cv2.setNumThreads(1)
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
    if enc.returncode != 0 or dec.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (decoder {dec.returncode}, encoder {enc.returncode})")
    # grade measured on the ACTUAL rendered footage area
    got = []
    for j in range(6):
        tt = (j + 0.5) * (n / fps) / 6
        r = media.run([media.ff(), "-v", "error", "-ss", f"{tt:.3f}", "-i", str(out), "-frames:v", "1", "-vf",
                       f"crop={W}:{VH}:0:{g.video_y},scale=480:-2,format=bgr24", "-f", "rawvideo", "-"])
        got.append(look.metrics(np.frombuffer(r.stdout, np.uint8).reshape(-1, 480, 3)))
    after_real = {k: round(float(np.mean([m[k] for m in got])), 2) for k in got[0]}
    rep["grade"]["after"] = after_real
    rep["grade"]["within_10pct"] = look.within(after_real, target)
    rep["timings"] = T
    eff_cx = (strip_x + pivot * CW) / src_w
    rep["_vgeom"] = {"W": W, "H": H, "video_y": g.video_y, "video_h": g.video_h, "cx": eff_cx.tolist(),
                     "zoom": cam["zoom"].tolist(), "frame0": span.fa,
                     "segmap": seg_map(segs)}
    return rep



def _render_ffmpeg(rep, T, cfg, q, g, W, H, fps, n, segs, norm, tmp, out, voc_wav, cam, CW, CH, src_w, src_h,
                   params, canvas0, ov, span, target, log):
    """Whole Short in ONE ffmpeg process / filter graph (no raw-frame pipes):
    trim+concat -> per-frame crop+zoom (sendcmd) -> scale -> sharpen -> 3D LUT grade -> clarity (grain extract/merge on
    luma) -> highlight glow (screen on luma) -> overlay on the white title canvas -> emoji sticker -> x264 + voice."""
    VW, VH = g.W, g.video_h
    z = cam["zoom"]
    cw = np.maximum(2, (np.round(CW / z / 2) * 2)).astype(int)
    ch = np.maximum(2, (np.round(CH / z / 2) * 2)).astype(int)
    cx_px = cam["cx"] * src_w
    x = np.clip(np.round(cx_px - cw / 2), 0, src_w - cw).astype(int)
    y = ((CH - ch) // 2).astype(int)
    with open(tmp / "cam.cmd", "w", encoding="ascii", newline="\n") as f:
        last = None
        for i in range(n):
            cur = (int(cw[i]), int(ch[i]), int(x[i]), int(y[i]))
            if cur != last:
                tt = max(0.0, (i - 0.5) / fps)
                f.write(f"{tt:.6f} crop@cam w {cur[0]}, crop@cam h {cur[1]}, crop@cam x {cur[2]}, crop@cam y {cur[3]};\n")
                last = cur
    cv2.imwrite(str(tmp / "canvas.png"), canvas0)
    st_args, st_graph = [], ""
    if ov.sticker is not None:
        rgba = ov.sticker
        cv2.imwrite(str(tmp / "sticker.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        sh, sw = rgba.shape[:2]
        sy = int(g.video_y + VH * cfg["layout"]["sticker_y_frac"] - sh / 2)
        st_args = ["-loop", "1", "-framerate", str(fps), "-i", "sticker.png"]
        st_graph = (f";[3:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1[st];"
                    f"[base][st]overlay=x={(W - sw) // 2}:y={sy}:shortest=1:format=auto[comp]")
    PW = int(np.ceil(VW / 32) * 32)               # padded width for the blend filters (SIMD tail bug otherwise)
    T_ = int(params.glow_thr * 255)
    sig = max(0.6, 0.024 * VW / 16)
    gsig = max(1.5, VW / 8 / 30)
    vg = cut.multi_trim_graph(segs, vout="vt")
    vg += (f";[vt]sendcmd=f=cam.cmd,crop@cam=w={cw[0]}:h={ch[0]}:x={x[0]}:y={y[0]},"
           f"scale={VW}:{VH}:flags=lanczos,setsar=1,unsharp=5:5:{params.sharpen:.4f}:3:3:0,"
           f"pad={PW}:{VH}:0:0:color=black,"          # swscale yuv->gbrp + blend SIMD leave the last 8 px of a 1080 row unset
           f"format=gbrp,lut3d=file=grade.cube:interp=trilinear,format=yuv420p,split=3[g0][g1][g2];"
           f"[g1]scale={max(2, PW // 16)}:{max(2, VH // 16)}:flags=area,gblur=sigma={sig:.3f},"
           f"scale={PW}:{VH}:flags=bilinear[bl];"
           f"[g0][bl]blend=c0_mode=grainextract:c1_mode=normal:c2_mode=normal,"
           f"lutyuv=y='clip((val-128)*{params.clarity:.4f}+128,0,255)'[hp];"      # high-pass x clarity (any amount)
           f"[g2][hp]blend=c0_mode=grainmerge:c0_opacity=1:c1_opacity=1:c2_opacity=1,split[c1][c2];"
           f"[c2]scale={max(2, PW // 8)}:{max(2, VH // 8)}:flags=area,"
           f"lutyuv=y='clip((val-{T_})*255/{max(1, 255 - T_)},0,255)':u=128:v=128,gblur=sigma={gsig:.3f},"
           f"scale={PW}:{VH}:flags=bilinear[gl];"
           f"[c1][gl]blend=c0_mode=screen:c0_opacity={params.glow:.4f}:c1_opacity=1:c2_opacity=1,crop={VW}:{VH}:0:0[foot];"
           f"[2:v]format=yuv420p[cv];[cv][foot]overlay=0:{g.video_y}:shortest=1[base]")
    vg += st_graph if st_graph else ";[base]null[comp]"
    vg += ";[comp]format=yuv420p[vout]"
    t = time.perf_counter()
    args = [media.ff(), "-v", "error", "-y",
            *cut.seek_args(segs[0]), "-i", str(Path(norm).resolve()),
            "-i", str(Path(voc_wav).resolve()),
            "-loop", "1", "-framerate", str(fps), "-i", "canvas.png", *st_args,
            "-filter_complex_threads", str(cfg["render"].get("decoder_filter_threads", 4)),
            "-filter_complex", vg, "-map", "[vout]", "-map", "1:a", "-frames:v", str(n),
            *cut.encode_args(q, fps, cfg["render"]["x264_threads"]), str(Path(out).resolve())]
    media.run(args, cwd=str(tmp))
    T["frames"] = round(time.perf_counter() - t, 1)
    written = media.count_frames(out)
    rep["frames_written"] = written
    rep["render_fps"] = round(written / max(1e-6, T["frames"]), 2)
    if written != n:
        raise RuntimeError(f"rendered {written} frames, expected {n} - refusing to write a drifting Short")
    got = []
    for j in range(6):
        tt = (j + 0.5) * (n / fps) / 6
        r = media.run([media.ff(), "-v", "error", "-ss", f"{tt:.3f}", "-i", str(out), "-frames:v", "1", "-vf",
                       f"crop={W}:{VH}:0:{g.video_y},scale=480:-2,format=bgr24", "-f", "rawvideo", "-"])
        got.append(look.metrics(np.frombuffer(r.stdout, np.uint8).reshape(-1, 480, 3)))
    after_real = {k: round(float(np.mean([m[k] for m in got])), 2) for k in got[0]}
    rep["grade"]["after"] = after_real
    rep["grade"]["within_10pct"] = look.within(after_real, target)
    rep["timings"] = T
    eff_cx = (x + cw / 2) / src_w
    rep["_vgeom"] = {"W": W, "H": H, "video_y": g.video_y, "video_h": g.video_h, "cx": eff_cx.tolist(),
                     "zoom": z.tolist(), "frame0": span.fa, "segmap": seg_map(segs), "src_w": src_w, "src_h": src_h}
    return rep
