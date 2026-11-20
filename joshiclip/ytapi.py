"""YouTube Data API v3 (API key, read-only): listings, view counts and comments without touching yt-dlp.

Why: every yt-dlp metadata request counts toward YouTube's "Sign in to confirm you're not a bot" limit. With the API
doing all lookups, yt-dlp is only used for what the API cannot do: downloading video / audio / captions of another
channel's videos (the API has no endpoint for that).

Key: YOUTUBE_API_KEY (or YT_API_KEY) from the environment / GitHub secret / .env. No key -> every function returns
None and callers fall back to yt-dlp. Costs: 1 quota unit per request (out of 10,000/day, shared with uploads at
1,600 each when the key is in the same Google Cloud project).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/youtube/v3/"


def key() -> str | None:
    from .tools import _dotenv
    env = _dotenv()
    for k in ("YOUTUBE_API_KEY", "YT_API_KEY"):
        v = os.environ.get(k) or env.get(k)
        if v:
            return v
    return None


def get(method: str, **params) -> dict | None:
    """One API call with retries on 5xx / network errors. None = no key or the API refused (caller falls back)."""
    k = key()
    if not k:
        return None
    url = API + method + "?" + urllib.parse.urlencode({**params, "key": k})
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < 2:
                time.sleep(3 * (attempt + 1)); continue
            body = e.read().decode(errors="replace")
            reason = re.search(r'"reason":\s*"(\w+)"', body)
            print(f"    YouTube API {method}: {e.code} {reason.group(1) if reason else ''} -> falling back to yt-dlp")
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(3 * (attempt + 1)); continue
            return None
    return None


def channel_id(handle_or_url: str) -> str | None:
    """'@souravjocshorts', 'https://www.youtube.com/@x/shorts' or 'UC...' -> 'UC...'."""
    m = re.search(r"(UC[\w-]{22})", handle_or_url)
    if m:
        return m.group(1)
    m = re.search(r"@([\w.-]+)", handle_or_url)
    if not m:
        return None
    r = get("channels", part="id", forHandle="@" + m.group(1))
    items = (r or {}).get("items") or []
    return items[0]["id"] if items else None


def playlist(playlist_id: str, n: int) -> list[dict] | None:
    """Newest first: [{vid, title}] (UULF<id> = long videos, UUSH<id> = Shorts)."""
    out, token = [], None
    while len(out) < n:
        p = {"part": "snippet", "playlistId": playlist_id, "maxResults": min(50, n - len(out))}
        if token:
            p["pageToken"] = token
        r = get("playlistItems", **p)
        if r is None:
            return None if not out else out
        for it in r.get("items") or []:
            sn = it["snippet"]
            out.append({"vid": sn["resourceId"]["videoId"], "title": sn.get("title", ""),
                        "published": sn.get("publishedAt")})
        token = r.get("nextPageToken")
        if not token:
            break
    return out


def _iso_dur(s: str) -> float | None:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return None
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + se)


def videos(ids: list[str]) -> dict[str, dict] | None:
    """{vid: {title, views, likes, comments, duration, published}} in batches of 50."""
    out = {}
    for i in range(0, len(ids), 50):
        r = get("videos", part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50]))
        if r is None:
            return None
        for it in r.get("items") or []:
            st = it.get("statistics", {})
            out[it["id"]] = {"title": it["snippet"].get("title", ""), "published": it["snippet"].get("publishedAt"),
                             "views": int(st["viewCount"]) if "viewCount" in st else None,
                             "likes": int(st["likeCount"]) if "likeCount" in st else None,
                             "comments": int(st["commentCount"]) if "commentCount" in st else None,
                             "duration": _iso_dur(it.get("contentDetails", {}).get("duration", ""))}
    return out


def comments(vid: str, n: int = 600) -> list[dict] | None:
    """Top-level comments, most relevant first (like yt-dlp's comment_sort=top): [{text, likes}]."""
    out, token = [], None
    while len(out) < n:
        p = {"part": "snippet", "videoId": vid, "maxResults": 100, "order": "relevance", "textFormat": "plainText"}
        if token:
            p["pageToken"] = token
        r = get("commentThreads", **p)
        if r is None:
            return None if not out else out
        for it in r.get("items") or []:
            c = it["snippet"]["topLevelComment"]["snippet"]
            out.append({"text": c.get("textOriginal") or c.get("textDisplay", ""), "likes": int(c.get("likeCount") or 0)})
        token = r.get("nextPageToken")
        if not token:
            break
    return out[:n]
