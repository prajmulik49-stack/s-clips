"""ffprobe / ffmpeg helpers shared by every stage."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction

import numpy as np

from . import tools


def ff() -> str:
    p = tools.ffmpeg().path
    if not p:
        raise RuntimeError("ffmpeg not found - run `joshiclip doctor`")
    return p


def fprobe() -> str:
    p = tools.ffprobe().path
    if not p:
        raise RuntimeError("ffprobe not found - run `joshiclip doctor`")
    return p


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(args, capture_output=True, env=tools.child_env(), **kw)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "ignore") if isinstance(r.stderr, bytes) else r.stderr
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, args[:12]))} ...\n{err[-3000:]}")
    return r


def probe(path) -> dict:
    r = run([fprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    d = json.loads(r.stdout)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    out = {"duration": float(d["format"].get("duration", 0)), "size": int(d["format"].get("size", 0)),
           "format": d["format"].get("format_name")}
    if v:
        out["video"] = {"codec": v["codec_name"], "width": int(v["width"]), "height": int(v["height"]),
                        "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
                        "fps": float(Fraction(v.get("r_frame_rate", "0/1"))) if v.get("r_frame_rate") != "0/0" else 0.0,
                        "avg_fps": float(Fraction(v["avg_frame_rate"])) if v.get("avg_frame_rate") not in (None, "0/0") else 0.0,
                        "nb_frames": int(v["nb_frames"]) if v.get("nb_frames", "").isdigit() else None,
                        "start_time": float(v.get("start_time", 0) or 0), "duration": float(v.get("duration", 0) or 0),
                        "pix_fmt": v.get("pix_fmt"), "color_transfer": v.get("color_transfer")}
    if a:
        out["audio"] = {"codec": a["codec_name"], "sample_rate": int(a["sample_rate"]), "channels": int(a["channels"]),
                        "start_time": float(a.get("start_time", 0) or 0), "duration": float(a.get("duration", 0) or 0),
                        "nb_samples": None}
    return out


def count_frames(path) -> int:
    """Exact decoded video frame count."""
    r = run([fprobe(), "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_packets", "-of", "csv=p=0", str(path)], text=True)
    return int(r.stdout.strip().split(",")[0])


def frame_pts(path) -> np.ndarray:
    """Presentation timestamps (s) of every video packet, sorted."""
    r = run([fprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
             "-of", "csv=p=0", str(path)], text=True)
    vals = [float(x) for x in r.stdout.split() if x and x != "N/A"]
    return np.sort(np.array(vals))


def read_audio(path, sr: int = 16000, mono: bool = True, t0: float | None = None, dur: float | None = None) -> np.ndarray:
    a = [ff(), "-v", "error"]
    if t0 is not None:
        a += ["-ss", f"{max(0.0, t0):.6f}"]
    if dur is not None:
        a += ["-t", f"{dur:.6f}"]
    a += ["-i", str(path), "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "f32le", "-"]
    x = np.frombuffer(run(a).stdout, np.float32)
    return x if mono else x.reshape(-1, 2)


def read_gray(path, t0: float, dur: float, fps: float, w: int = 160, h: int = 90) -> np.ndarray:
    a = [ff(), "-v", "error", "-ss", f"{max(0.0, t0):.6f}", "-t", f"{dur:.6f}", "-i", str(path), "-an",
         "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    return np.frombuffer(run(a).stdout, np.uint8).reshape(-1, h, w).astype(np.float32)


def audio_samples(path, sr: int | None = None) -> int:
    """Exact decoded audio sample count (per channel)."""
    p = probe(path)
    sr = sr or p["audio"]["sample_rate"]
    r = run([ff(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"])
    return len(r.stdout) // 2
