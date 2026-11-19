"""Subtle face-follow virtual camera (deterministic).

Faces come from the 2 fps analysis (normalized boxes). Target = largest face, or the union of the two largest
when both are talking-size and fit. A dead-zone keeps the frame still for small moves; a critically damped
spring gives slow, jerk-free easing; hard scene cuts snap. Zoom stays in [1.00, zoom_max].
"""
from __future__ import annotations

import numpy as np


def path(vision: dict, span_start: float, n_frames: int, fps: int, src_aspect: float, out_aspect: float,
         scene_cuts: list[float], peak_t: float | None, cfg: dict, times: np.ndarray | None = None) -> dict:
    """times: analysis-timeline time of each output frame (jump cuts make it non-contiguous); default contiguous."""
    if times is None:
        times = span_start + np.arange(n_frames) / fps
    vf = float(vision["fps"])
    faces = vision["faces"]                              # [N, 4, 5] x, y, w, h, score (normalized)
    base_w = out_aspect / src_aspect                     # crop width (fraction of source width) at zoom 1
    zmin, zmax = cfg["zoom_min"], cfg["zoom_max"]
    targets = np.full(n_frames, 0.5, np.float64)
    has = np.zeros(n_frames, bool)
    last = 0.5
    for i in range(n_frames):
        t = float(times[i])
        k = int(round(t * vf))
        if 0 <= k < len(faces):
            fs = [f for f in faces[k] if f[2] > 0.01]
            if fs:
                fs.sort(key=lambda f: (-round(float(f[2] * f[3]), 6), float(f[0])))
                a = fs[0]
                cx = a[0] + a[2] / 2
                if len(fs) > 1 and fs[1][2] * fs[1][3] > 0.35 * a[2] * a[3]:
                    x0 = min(a[0], fs[1][0]); x1 = max(a[0] + a[2], fs[1][0] + fs[1][2])
                    if x1 - x0 < base_w / zmin * 0.9:
                        cx = (x0 + x1) / 2
                last = float(cx); has[i] = True
        targets[i] = last
    # dead-zone + critically damped spring (omega rad/s); snap on scene cuts
    cut_frames = set()
    for c in scene_cuts:                                   # first output frame at/after each scene cut
        j = int(np.searchsorted(times, c))
        if 0 < j < n_frames and times[j - 1] < c:
            cut_frames.add(j)
    jumps = np.flatnonzero(np.diff(times) > 1.5 / fps) + 1   # jump cuts also snap the camera
    cut_frames.update(int(j) for j in jumps)
    w = cfg["omega"]
    dt = 1.0 / fps
    x = targets[0]; v = 0.0
    goal = targets[0]
    xs = np.zeros(n_frames)
    dz = cfg["deadzone"] * base_w
    for i in range(n_frames):
        if i in cut_frames:
            x = targets[i]; v = 0.0; goal = targets[i]
        if abs(targets[i] - goal) > dz:
            goal = targets[i]
        a = w * w * (goal - x) - 2 * w * v
        v += a * dt
        x += v * dt
        xs[i] = x
    # zoom: gentle push-in towards the peak, eased
    z = np.full(n_frames, zmin)
    if peak_t is not None:
        bump = np.exp(-0.5 * ((times - peak_t) / cfg["zoom_sigma_s"]) ** 2)
        z = zmin + (zmax - zmin) * bump
    cw = base_w / z
    xs = np.clip(xs, cw / 2, 1 - cw / 2)
    return {"cx": xs, "zoom": z, "crop_w": cw, "face_frames": float(has.mean())}
