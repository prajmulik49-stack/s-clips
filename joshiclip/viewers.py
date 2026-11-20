"""Viewer signal from the vlog's top YouTube comments ("2:40 piyush 😂😂😂").

Fans timestamp the moments they loved. That is the only *audience* signal we have before posting, and it is free:
yt-dlp reads the top ~600 comments in ~30 s (metadata only). On 4H_oOhnN6Dc, 600 comments held 198 timestamps.

fetch()     top-N comments -> work/<vid>/comments.json (cached; workers never touch the network)
hits()      every timestamp a real (non-spam) comment points at, weighted by likes
density()   per-second curve -> comments_per_sec.npy (pipeline already feeds it to peaks + scoring)
hotspots()  strongest comment clusters -> pseudo peaks, so a moment viewers point at becomes a Short candidate
score()     0..1 viewer demand for a span
title()     the most-liked clean viewer line about the clip ("Sahil Ka Muh Dekhne Layak Hai 😂")
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")
# like-bait / greetings / devotional one-liners that get pasted onto any timestamp
SPAM = ("padh liya", "padh lia", "kasam", "like nhi", "like nahi", "like na", "like kar", "like karo", "like button",
        "respect button", "no like", "fail", "exam", "pass section", "first comment", "my first", "first view", "vote",
        "subscribe", "sub kar", "good morning", "good night", "radhe", "jai shri", "jay shri", "jai shree", "ram ram",
        "kis kis", "kon kon", "kaun kaun", "who is watching", "anyone", "pin karo", "pin kar", "reply", "heart de",
        "http", "www.", "@", "राधे", "जय श्री", "राम", "comment section", "gta 6", "gta6")
# fine as a vote for the moment, never as a burned-in title
UNSAFE = ("nasha", "nashe", "ganja", "drug", "daru", "daaru", "sharab", "chutiya", "chutia", "gandu", "bc ", "mc ",
          "madar", "bhosd", "kamina", "kamine", "harami", "sexy", "hot ", "kiss", "divorce", "death", "mar gaya", "mar gayi",
          "maut", "suicide", "gali", "gaali", "galli", "abuse", "swear", "slap", "thappad", "fake", "scripted", "paid", "boring", "bakwas", "ghatiya", "andha", "pagal", "gadha", "ullu")
GENERIC = {"best moment", "best part", "funny moment", "nice", "wow", "lol", "omg", "superb", "amazing", "cute", "love",
           "beautiful", "ky view", "kya view", "nice view", "painting lag rahi hai"}
GENERIC_PARTS = (" best part", " best moment", " this video", " this vlog", " funniest part", " funny part",
                 " my favourite", " my favorite", " fav part", " sabse funny", " sabse best")
PHRASE_FIX = {"oh my good": "oh my god", "ho gya": "ho gaya", "hogaya": "ho gaya", "hogya": "ho gaya"}
FIX = {"rha": "raha", "rhi": "rahi", "rhe": "rahe", "nhi": "nahi", "nai": "nahi", "kese": "kaise", "ky": "kya",
       "h": "hai", "hy": "hai", "gya": "gaya", "gyi": "gayi", "bhot": "bahut", "bht": "bahut", "mst": "mast",
       "dkho": "dekho", "lg": "lag", "kr": "kar", "krna": "karna", "ni": "nahi", "pr": "par", "yr": "yaar", "yar": "yaar"}
TITLE_EMOJI = "😂🤣😱😳😮😍🥺😅🔥😢😭💃❤🤯😆🙈"
EMOJI_RX = re.compile("[\U0001F300-\U0001FAFF☀-➿❤️‍]+")


# ---------------------------------------------------------------- fetch
def fetch(cfg, url: str, wd: Path, log=print) -> Path | None:
    v = cfg.get("viewers", {})
    p = wd / "comments.json"
    if p.exists():
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h < float(v.get("refresh_hours", 12)) or not v.get("allow_network", True):
            return p
    if not v.get("allow_network", True):
        return None
    from . import ingest, tools, ytapi
    n = int(v.get("max_comments", 600))
    t = time.perf_counter()
    vid = ingest.video_id_for(url)
    cs = ytapi.comments(vid, n)                      # API first: ~6 quota units, no yt-dlp request
    if cs is not None:
        meta = (ytapi.videos([vid]) or {}).get(vid, {})
        p.write_text(json.dumps({"url": url, "fetched": int(time.time()), "duration": meta.get("duration"),
                                 "view_count": meta.get("views"), "title": meta.get("title"),
                                 "upload_date": (meta.get("published") or "")[:10].replace("-", ""),
                                 "comments": cs, "source": "api"}, ensure_ascii=False), encoding="utf-8")
        log(f"    {len(cs)} top comments in {time.perf_counter() - t:.0f}s (YouTube API)")
        return p
    args = ingest.ytdlp_base(cfg) + ["--dump-json", "--skip-download", "--write-comments", "--extractor-args",
                                     f"youtube:max_comments={n},all,0,0;comment_sort=top", url]
    t = time.perf_counter()
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=tools.child_env(), timeout=int(v.get("timeout_s", 240)))
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - viewers are a bonus signal; never fail a run for them
        log(f"    viewer comments unavailable ({type(e).__name__}); scoring without them")
        return p if p.exists() else None
    cs = [{"text": c.get("text", ""), "likes": int(c.get("like_count") or 0)} for c in d.get("comments") or []
          if c.get("parent", "root") == "root"]
    p.write_text(json.dumps({"url": url, "fetched": int(time.time()), "duration": d.get("duration"),
                             "view_count": d.get("view_count"), "title": d.get("title"), "upload_date": d.get("upload_date"),
                             "comments": cs}, ensure_ascii=False), encoding="utf-8")
    log(f"    {len(cs)} top comments in {time.perf_counter() - t:.0f}s")
    return p


# ---------------------------------------------------------------- parse
def _is_spam(low: str) -> bool:
    return any(s in low for s in SPAM)


def hits(comments: list[dict], duration: float, min_t: float = 20.0) -> list[dict]:
    out = []
    for c in comments:
        text = c.get("text", "")
        ts = []
        for m in TS.finditer(text):
            h, mm, ss = m.groups()
            if int(ss) >= 60 or (h and int(mm) >= 60):
                continue
            s = int(h or 0) * 3600 + int(mm) * 60 + int(ss)
            if min_t <= s <= duration:
                ts.append(s)
        if not ts or _is_spam(text.lower()):
            continue
        w = (1.0 + 0.5 * math.log1p(min(c.get("likes", 0), 1000))) / len(ts)    # a list of 5 times = 5 weak votes
        for s in ts:
            out.append({"t": float(s), "w": round(w, 4), "likes": c.get("likes", 0), "n_ts": len(ts), "text": text})
    out.sort(key=lambda h: (h["t"], -h["likes"]))
    return out


def density(hs: list[dict], n_sec: int, shift_s: float = 2.0, sigma_s: float = 3.0) -> np.ndarray:
    """Comments name the second a moment *starts*; the payoff is a little later -> shift +2 s, blur 3 s."""
    x = np.zeros(n_sec, np.float32)
    k = np.arange(-int(3 * sigma_s), int(3 * sigma_s) + 1)
    g = np.exp(-0.5 * (k / sigma_s) ** 2).astype(np.float32)
    for h in hs:
        c = int(round(h["t"] + shift_s))
        for kk, gg in zip(k, g):
            if 0 <= c + kk < n_sec:
                x[c + kk] += h["w"] * gg
    return x


def build(cfg, url: str | None, wd: Path, duration: float, log=print) -> np.ndarray | None:
    """fetch -> hits -> density; writes comments_per_sec.npy + viewer_hits.json next to the analysis."""
    if not url or not cfg.get("viewers", {}).get("enabled", False):
        return None
    p = fetch(cfg, url, wd, log)
    if p is None:
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    hs = hits(d["comments"], duration, float(cfg["viewers"].get("min_t_s", 20)))
    n_sec = int(math.ceil(duration))
    x = density(hs, n_sec)
    old = wd / "comments_per_sec.npy"
    if not (old.exists() and np.array_equal(np.load(old), x)):      # keep the file (and stage caches) stable
        np.save(old, x)
    (wd / "viewer_hits.json").write_text(json.dumps(hs, ensure_ascii=False), encoding="utf-8")
    log(f"    viewer timestamps: {len(hs)} from {len(d['comments'])} comments")
    return x


def vlog_title(wd: Path) -> str | None:
    p = Path(wd) / "comments.json"
    return json.loads(p.read_text(encoding="utf-8")).get("title") if p.exists() else None


def load_hits(A: dict) -> list[dict]:
    if "_viewer_hits" not in A:
        p = Path(A["workdir"]) / "viewer_hits.json" if A.get("workdir") else None
        A["_viewer_hits"] = json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else []
    return A["_viewer_hits"]


# ---------------------------------------------------------------- use
def hotspots(A: dict, cfg) -> list[dict]:
    """Comment clusters (>= min_comments distinct comments, >= min_weight within +-6 s) -> pseudo peaks."""
    x = A.get("comments_per_sec")
    v = cfg.get("viewers", {})
    if x is None or not np.sum(x):
        return []
    hs = load_hits(A)
    must = float(cfg["engine"].get("must_peak_strength", 3.0))
    out, taken = [], np.zeros(len(x), bool)
    for t in sorted(range(len(x)), key=lambda i: (-round(float(x[i]), 4), i)):
        if x[t] <= 0:
            break
        if taken[t]:
            continue
        near = [h for h in hs if abs(h["t"] + 2 - t) <= 6]
        wt = sum(h["w"] for h in near)
        if len({h["text"] for h in near}) >= int(v.get("hotspot_min_comments", 3)) and wt >= float(v.get("hotspot_min_weight", 5.0)):
            tc = sum(h["w"] * h["t"] for h in near) / wt + 1.0    # like-weighted time viewers name (+1 s into the moment)
            out.append({"id": f"v{t}", "t": round(float(tc), 2), "type": "viewers", "strength": round(must + math.log2(wt / 5.0 + 1), 3),
                        "comments": len(near)})
            taken[max(0, t - 20):t + 20] = True
    return out


def score(A: dict, a: float, b: float, window_s: int = 45) -> float:
    """Weight of the timestamps that point INSIDE [a, b] (>= 3 s before the end, so the moment itself is shown),
    relative to the best 45 s window of the video (0..1). A clip that starts right after the moment scores ~0."""
    hs = load_hits(A)
    if not hs:
        return 0.0
    if "_viewer_best" not in A:
        n = int(max(h["t"] for h in hs)) + window_s + 2
        raw = np.zeros(n, np.float32)
        for h in hs:
            raw[int(h["t"])] += h["w"]
        A["_viewer_best"] = float(np.convolve(raw, np.ones(window_s), "valid").max(initial=0)) + 1e-9
    got = sum(h["w"] for h in hs if a - 1.0 <= h["t"] <= b - 3.0)
    return float(np.clip(got / A["_viewer_best"], 0, 1))


def _clean(text: str) -> tuple[str, str]:
    emo = "".join(dict.fromkeys(ch for ch in "".join(EMOJI_RX.findall(text)) if ch in TITLE_EMOJI))[:2]
    t = TS.sub(" ", text)
    t = EMOJI_RX.sub(" ", t)
    t = re.sub(r"[‘’`´]", "'", t)
    t = re.sub(r"[^A-Za-z0-9?!' ]+", " ", t)
    words = [FIX.get(w.lower(), w.lower()) for w in t.split()]
    out = " ".join(words)
    for k, v in PHRASE_FIX.items():
        out = re.sub(rf"\b{k}\b", v, out)
    return out, emo


def title(A: dict, a: float, b: float, used: set, names: list[str], min_likes: int = 3, max_chars: int = 34) -> str | None:
    """Best viewer line about [a, b]: one timestamp inside the clip, Latin script, 3-8 words, liked by >= min_likes."""
    best, bkey = None, None
    name_map = {n.lower(): n for n in names}
    for h in load_hits(A):
        if h["n_ts"] != 1 or h["likes"] < min_likes or not (a - 2 <= h["t"] <= b - 3):
            continue
        raw = h["text"]
        letters = re.findall(r"[^\W\d_]", TS.sub("", raw))
        if not letters or sum(1 for ch in letters if ch.isascii()) / len(letters) < 0.9:
            continue                                     # Devanagari etc. -> the header font can't draw it
        txt, emo = _clean(raw)
        low = f" {txt} "
        if any(u in low for u in UNSAFE) or txt.strip("?! ") in GENERIC or any(g in low for g in GENERIC_PARTS):
            continue
        ws = txt.split()
        content = [w for w in ws if len(w) >= 3]
        if not (3 <= len(ws) <= 8) or len(content) < 2 or len(txt) > max_chars:
            continue
        t_out = " ".join(name_map.get(w, w[:1].upper() + w[1:]) for w in ws)
        t_out = f"{t_out} {emo or '😄'}"
        if t_out in used:
            continue
        has_name = any(w in name_map for w in ws)
        k = (round(math.log1p(h["likes"]) + (0.5 if has_name else 0) + (0.2 if emo else 0), 6), -h["t"])
        if bkey is None or k > bkey:
            best, bkey = t_out, k
    return best
