"""Resolve external binaries once and hand the same absolute paths to every stage.

Why: on this machine the VS Code process environment predates the ffmpeg install, so
child processes (Python, yt-dlp, cmd) inherit a PATH without ffmpeg even though the
registry PATH and Git-Bash have it. We therefore never rely on a bare "ffmpeg" lookup:
config override -> PATH -> registry PATH (Windows) -> well-known install dirs, and we
prefer the build that has every filter/encoder the engine needs.
"""
from __future__ import annotations

import functools
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

REQUIRED_FILTERS = ["trim", "atrim", "setpts", "asetpts", "loudnorm", "ebur128", "sidechaincompress",
                    "equalizer", "acompressor", "alimiter", "unsharp", "scale", "aresample"]
NICE_FILTERS = ["libplacebo", "vidstabdetect", "rubberband"]
NICE_ENCODERS = ["libx264", "libx265", "hevc_qsv", "h264_qsv", "libsvtav1"]


@dataclass
class Tool:
    name: str
    path: str | None
    version: str = ""
    source: str = ""
    missing_filters: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def _exe(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def _registry_path_dirs() -> list[str]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []
    out = []
    for root, key in [(winreg.HKEY_CURRENT_USER, r"Environment"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]:
        try:
            with winreg.OpenKey(root, key) as k:
                val, _ = winreg.QueryValueEx(k, "Path")
                out += [os.path.expandvars(p) for p in val.split(";") if p]
        except OSError:
            pass
    return out


def _known_dirs(name: str) -> list[str]:
    if os.name != "nt":
        return ["/usr/local/bin", "/usr/bin", "/opt/homebrew/bin", "/snap/bin"]
    home = os.path.expanduser("~")
    pats = [os.path.join(home, r"AppData\Local\Microsoft\WinGet\Packages\*FFmpeg*\*\bin"),
            os.path.join(home, r"Downloads\ffmpeg*\*\bin"), os.path.join(home, r"Downloads\ffmpeg*\bin"),
            r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin", os.path.join(home, r"scoop\shims"),
            r"C:\ProgramData\chocolatey\bin",
            os.path.join(home, r"Downloads\chromaprint*"), os.path.join(home, r"Downloads\chromaprint*\*")]
    return sorted({d for p in pats for d in glob.glob(p)}, reverse=True)


def _candidates(name: str, override: str | None) -> list[tuple[str, str]]:
    c = []
    if override:
        c.append((override, "config"))
    w = shutil.which(name)
    if w:
        c.append((w, "PATH"))
    for d in _registry_path_dirs():
        p = os.path.join(d, _exe(name))
        if os.path.isfile(p):
            c.append((p, "registry PATH"))
    for d in _known_dirs(name):
        p = os.path.join(d, _exe(name))
        if os.path.isfile(p):
            c.append((p, "known location"))
    seen, out = set(), []
    for p, s in c:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen and os.path.isfile(p):
            seen.add(k); out.append((os.path.abspath(p), s))
    return out


def _run(args: list[str]) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=30)
        return r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _inspect_ffmpeg(path: str) -> Tool:
    ver = (_run([path, "-hide_banner", "-version"]).splitlines() or [""])[0]
    flt = _run([path, "-hide_banner", "-filters"])
    enc = _run([path, "-hide_banner", "-encoders"])
    have = set(re.findall(r"^\s*[TSC.|]{2,3}\s+(\w+)", flt, re.M))
    t = Tool("ffmpeg", path, ver)
    t.missing_filters = [f for f in REQUIRED_FILTERS if f not in have]
    t.extras = {"nice_filters": [f for f in NICE_FILTERS if f in have],
                "encoders": [e for e in NICE_ENCODERS if re.search(rf"\b{e}\b", enc)]}
    return t


def _score(t: Tool) -> tuple:
    return (-len(t.missing_filters), len(t.extras.get("nice_filters", [])), len(t.extras.get("encoders", [])), t.version)


@functools.lru_cache(maxsize=None)
def ffmpeg(override: str | None = None) -> Tool:
    """Best ffmpeg build available (fewest missing filters, most extras)."""
    best = None
    for p, src in _candidates("ffmpeg", override):
        t = _inspect_ffmpeg(p); t.source = src
        if src == "config" and not t.missing_filters:
            return t
        if best is None or _score(t) > _score(best):
            best = t
    return best or Tool("ffmpeg", None)


@functools.lru_cache(maxsize=None)
def ffprobe(override: str | None = None) -> Tool:
    ff = ffmpeg()
    if ff.path:                                    # always pair ffprobe with the chosen ffmpeg build
        p = os.path.join(os.path.dirname(ff.path), _exe("ffprobe"))
        if os.path.isfile(p):
            return Tool("ffprobe", p, (_run([p, "-hide_banner", "-version"]).splitlines() or [""])[0], "next to ffmpeg")
    for p, src in _candidates("ffprobe", override):
        return Tool("ffprobe", p, (_run([p, "-hide_banner", "-version"]).splitlines() or [""])[0], src)
    return Tool("ffprobe", None)


@functools.lru_cache(maxsize=None)
def fpcalc(override: str | None = None) -> Tool:
    for p, src in _candidates("fpcalc", override):
        return Tool("fpcalc", p, _run([p, "-version"]).strip(), src)
    return Tool("fpcalc", None)


def ytdlp_opts() -> dict:
    """Options every yt-dlp call must carry (Python API)."""
    ff = ffmpeg()
    return {"ffmpeg_location": os.path.dirname(ff.path)} if ff.path else {}


def ytdlp_args() -> list[str]:
    """Same, for the yt-dlp CLI."""
    ff = ffmpeg()
    a = ["--ffmpeg-location", os.path.dirname(ff.path)] if ff.path else []
    env = _dotenv()
    ck = os.environ.get("JOSHICLIP_YTDLP_COOKIES") or env.get("JOSHICLIP_YTDLP_COOKIES")   # CI: datacenter IPs
    if not ck:
        local = Path(__file__).resolve().parents[1] / "cookies.txt"                       # git-ignored
        ck = str(local) if local.is_file() else None
    browser = os.environ.get("JOSHICLIP_YTDLP_BROWSER") or env.get("JOSHICLIP_YTDLP_BROWSER")   # e.g. firefox
    if ck and os.path.isfile(ck):
        a += ["--cookies", ck]
    elif browser and _browser_ready(browser):
        a += ["--cookies-from-browser", browser]
    return a


_WARNED: set = set()


def _browser_ready(browser: str) -> bool:
    """Only pass --cookies-from-browser when that browser has a profile; else warn once and go on without cookies
    (a missing Firefox profile used to crash the whole run)."""
    b = browser.split(":")[0].lower()
    if b != "firefox":
        return True
    roots = [Path(os.environ.get("APPDATA", "")) / "Mozilla/Firefox/Profiles",
             Path(os.environ.get("LOCALAPPDATA", "")) / "Packages/Mozilla.Firefox_n80bbvh6b1yt2/LocalCache/Roaming/Mozilla/Firefox/Profiles",
             Path.home() / ".mozilla/firefox"]
    if any(r.is_dir() and any(r.glob("*/cookies.sqlite")) for r in roots):
        return True
    if b not in _WARNED:
        _WARNED.add(b)
        print("  WARNING: JOSHICLIP_YTDLP_BROWSER=firefox but Firefox has no profile with cookies yet: open Firefox, "
              "log into youtube.com with a spare Google account, then run again. Continuing WITHOUT cookies.", flush=True)
    return False


def _dotenv() -> dict:
    """Project .env (local runs only; real environment variables win)."""
    p = Path(__file__).resolve().parents[1] / ".env"
    out = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def child_env() -> dict:
    """Environment for subprocesses: chosen ffmpeg dir first on PATH, UTF-8 I/O."""
    env = dict(os.environ)
    ff = ffmpeg()
    if ff.path:
        env["PATH"] = os.path.dirname(ff.path) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env
