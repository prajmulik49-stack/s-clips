"""Ingest: download (yt-dlp) -> normalize ONCE to CFR + 48 kHz -> 360p proxy + 16 kHz mono wav.

Every later stage reads only these files:
  work/<id>/norm.mp4            CFR (30 or 60 fps), H.264 CRF 16, FLAC 48 kHz (sample-exact, no priming delay)
  work/<id>/proxy_360p.mp4      analysis proxy (same timeline)
  work/<id>/audio_16k_mono.wav  analysis audio (same timeline)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import media, tools
from .config import Config

YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|/live/)([A-Za-z0-9_-]{11})")
NORM_VERSION = 1


def video_id_for(src: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    m = YT_ID.search(src)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", src):
        return src
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(src).stem)[:48]


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://")) or bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", s))


# ---------------------------------------------------------------- yt-dlp
_updated = False


def ytdlp_base(cfg: Config, proxy: bool = True) -> list[str]:
    d = cfg["download"]
    return [sys.executable, "-m", "yt_dlp", *tools.ytdlp_args(proxy), "--sleep-interval", str(d["sleep_interval"]),
            "--max-sleep-interval", str(d["max_sleep_interval"]), "--socket-timeout", str(d.get("socket_timeout", 20)),
            "--retries", "10", "--extractor-retries", "5", "--no-warnings"]


def ytdlp_update(cfg: Config, log=print) -> None:
    """`yt-dlp -U` (pip installs can't self-update, so fall back to pip -U)."""
    global _updated
    if _updated or not cfg["download"].get("update_first", True):
        return
    _updated = True
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "-U"], capture_output=True, text=True, env=tools.child_env())
    msg = (r.stdout + r.stderr).strip().splitlines()
    if any("pip" in m.lower() for m in msg):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp[default]"], capture_output=True, text=True)
        log(f"  yt-dlp updated via pip (rc={r.returncode})")
    else:
        log("  " + (msg[-1] if msg else "yt-dlp -U done"))


def latest_long_video(cfg: Config, channel_id: str | None = None) -> str:
    cid = channel_id or cfg["channel"]["official_id"]
    pl = "UULF" + cid[2:]
    from . import ytapi
    items = ytapi.playlist(pl, 1)                    # API first: no yt-dlp request -> no bot-check budget used
    if items:
        return items[0]["vid"]
    r = media.run(ytdlp_base(cfg) + ["--flat-playlist", "--playlist-end", "1", "--print", "id",
                                     f"https://www.youtube.com/playlist?list={pl}"], text=True)
    return r.stdout.split()[0]


def download(cfg: Config, url: str, log=print) -> Path:
    ytdlp_update(cfg, log)
    vid = video_id_for(url)
    out_dir = cfg.path("downloads") / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(p for p in out_dir.glob(f"{vid}.*") if p.suffix in (".mkv", ".mp4", ".webm") and ".part" not in p.name)
    if done:
        return done[0]
    if not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={url}"
    d = cfg["download"]
    media.run(ytdlp_base(cfg) + ["-f", d["format"], "--merge-output-format", d["merge_format"], "--continue",
                                 "--no-overwrites", "-o", str(out_dir / "%(id)s.%(ext)s"),
                                 "--write-info-json", url])
    done = sorted(p for p in out_dir.glob(f"{vid}.*") if p.suffix in (".mkv", ".mp4", ".webm"))
    if not done:
        raise RuntimeError(f"download produced no media file in {out_dir}")
    return done[0]


# ---------------------------------------------------------------- normalize
def file_fingerprint(path: Path) -> str:
    """size + sha256 of the first and last 4 MB: cheap, stable identity for multi-GB files."""
    st = path.stat()
    h = hashlib.sha256(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(0, st.st_size - (4 << 20)))
        h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def nominal_fps(info: dict, allowed: list[int]) -> int:
    v = info["video"]
    # avg_frame_rate describes VFR phone footage better than r_frame_rate (which can be 90000/1 etc.)
    f = v["avg_fps"] or v["fps"]
    return min(allowed, key=lambda a: (abs(a - f), a))


def _x264_args(n: dict, threads: int) -> list[str]:
    a = ["-c:v", n["video_codec"], "-crf", str(n["crf"]), "-preset", n["preset"], "-pix_fmt", "yuv420p"]
    if threads:
        a += ["-threads", str(threads)]
    return a


def normalize(cfg: Config, src: Path, workdir: Path, log=print) -> dict:
    """Idempotent: skips work when norm.json matches the source fingerprint + normalize config."""
    n = cfg["normalize"]
    workdir.mkdir(parents=True, exist_ok=True)
    meta_p = workdir / "norm.json"
    key = {"src": file_fingerprint(src), "cfg": n, "v": NORM_VERSION}
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("key") == key and all((workdir / f).exists() for f in ("norm.mp4", "proxy_360p.mp4", "audio_16k_mono.wav")):
            log("  [cache] normalize")
            return meta
    info = media.probe(src)
    if "video" not in info or "audio" not in info:
        raise RuntimeError(f"{src} needs one video and one audio stream")
    fps = nominal_fps(info, n["allowed_fps"])
    h_src = info["video"]["height"]
    h = min(h_src, n["max_height"])
    timings = {}

    # 1) norm.mp4 -- the ONLY file later stages cut from
    t = time.perf_counter()
    tmp = workdir / "norm.partial.mp4"
    vf = [f"fps={fps}"]
    if h != h_src:
        vf.append(f"scale=-2:{h}:flags=lanczos")
    vf.append("format=yuv420p")
    args = [media.ff(), "-v", "error", "-stats_period", "30", "-y"]
    if n.get("hwaccel_decode", "none") == "qsv":
        args += ["-hwaccel", "qsv"]
    args += ["-i", str(src), "-map", "0:v:0", "-map", "0:a:0", "-vf", ",".join(vf), "-fps_mode", "cfr",
             *_x264_args(n, n.get("x264_threads", 0)), "-g", str(2 * fps), "-keyint_min", str(fps),
             "-af", f"aresample={n['audio_rate']}:async=1:first_pts=0", "-ar", str(n["audio_rate"]),
             "-c:a", n["audio_codec"], "-map_metadata", "-1", "-movflags", "+faststart", str(tmp)]
    log(f"  normalizing {src.name} -> {fps} fps CFR, {h}p, {n['audio_rate']} Hz (this is the slow one-time step)")
    media.run(args)
    os.replace(tmp, workdir / "norm.mp4")
    timings["norm"] = round(time.perf_counter() - t, 1)

    # 2) proxy + analysis audio, both derived from norm.mp4 so they share its timeline
    t = time.perf_counter()
    pfps = fps if fps <= n["proxy_max_fps"] else fps // 2
    media.run([media.ff(), "-v", "error", "-y", "-i", str(workdir / "norm.mp4"), "-vf",
               f"fps={pfps},scale=-2:{n['proxy_height']}:flags=area,format=yuv420p", "-fps_mode", "cfr",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(n["proxy_crf"]), "-g", str(2 * pfps),
               "-c:a", "aac", "-b:a", "96k", "-map_metadata", "-1", str(workdir / "proxy_360p.mp4")])
    media.run([media.ff(), "-v", "error", "-y", "-i", str(workdir / "norm.mp4"), "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact", str(workdir / "audio_16k_mono.wav")])
    timings["proxy_audio"] = round(time.perf_counter() - t, 1)

    ni = media.probe(workdir / "norm.mp4")
    meta = {"key": key, "source": str(src), "source_info": info, "fps": fps, "proxy_fps": pfps,
            "width": ni["video"]["width"], "height": ni["video"]["height"], "duration": ni["duration"],
            "norm_info": ni, "timings_s": timings}
    meta["checks"] = verify_norm(workdir / "norm.mp4", fps)
    meta_p.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    return meta


def verify_norm(path: Path, fps: int) -> dict:
    """CFR + aligned stream starts + matching durations. Raises on failure."""
    info = media.probe(path)
    v, a = info["video"], info["audio"]
    pts = media.frame_pts(path)
    d = (pts[1:] - pts[:-1]) if len(pts) > 1 else pts
    res = {"r_frame_rate": v["r_frame_rate"], "avg_frame_rate": v["avg_frame_rate"],
           "frames": int(len(pts)), "frame_interval_ms": [round(float(d.min()) * 1000, 3), round(float(d.max()) * 1000, 3)],
           "v_start": v["start_time"], "a_start": a["start_time"], "sample_rate": a["sample_rate"],
           "v_dur": v["duration"], "a_dur": a["duration"]}
    ok = (abs(v["avg_fps"] - fps) < 1e-3 and abs(float(d.max()) - 1 / fps) < 0.5e-3 and abs(float(d.min()) - 1 / fps) < 0.5e-3
          and abs(v["start_time"] - a["start_time"]) < 0.5 / fps and a["sample_rate"] == 48000
          and abs(v["duration"] - a["duration"]) < 2.0 / fps + 0.03)
    res["ok"] = bool(ok)
    if not ok:
        raise RuntimeError(f"normalized file failed CFR/sync verification: {res}")
    return res


def resolve_source(cfg: Config, target: str | None, latest: bool, log=print) -> tuple[str, Path]:
    if latest or not target:
        ytdlp_update(cfg, log)
        target = latest_long_video(cfg)
        log(f"  newest long video: {target}")
    if os.path.exists(target):
        return video_id_for(target), Path(target).resolve()
    if is_url(target):
        return video_id_for(target), download(cfg, target, log)
    raise FileNotFoundError(target)
