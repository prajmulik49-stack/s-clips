"""Facebook Page Reels: every Short that went to YouTube is also posted to ONE Facebook Page, at the same time.

Never allowed to hurt YouTube: daily.post() calls post_reel() only AFTER the YouTube upload succeeded, and any
Facebook error is logged and ignored. Off unless FB_PAGE_ID + FB_PAGE_TOKEN exist (GitHub secrets / .env).

One-time setup (on the PC):  .env gets FB_APP_ID, FB_APP_SECRET and FB_USER_TOKEN (Graph API Explorer, permissions
pages_show_list + pages_read_engagement + pages_manage_posts), then  `python -m joshiclip fb-token`  turns it into a
Page token that never expires and writes FB_PAGE_ID / FB_PAGE_TOKEN into .env (add both as GitHub secrets).

Reels API: POST /{page}/video_reels upload_phase=start -> upload the bytes to rupload.facebook.com ->
upload_phase=finish with video_state PUBLISHED or SCHEDULED (+ scheduled_publish_time).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

GRAPH = "https://graph.facebook.com"


def _ver(cfg=None) -> str:
    return str(((cfg or {}).get("facebook") or {}).get("graph_version", "v23.0"))


def _env(k: str) -> str:
    from .youtube import load_env
    load_env()
    return (os.environ.get(k) or "").strip()


def enabled(cfg) -> bool:
    f = cfg.get("facebook") or {}
    return bool(f.get("enabled", True)) and bool(_env("FB_PAGE_ID")) and bool(_env("FB_PAGE_TOKEN"))


def _check(r) -> dict:
    try:
        j = r.json()
    except ValueError:
        j = {"raw": r.text[:300]}
    if r.status_code >= 400 or (isinstance(j, dict) and "error" in j):
        e = j.get("error", j) if isinstance(j, dict) else j
        raise RuntimeError(f"Facebook {r.status_code}: {str(e)[:300]}")
    return j


def caption(title: str, hashtags: list[str], credit: str | None, link: str | None = None) -> str:
    lines = [title.strip()]
    if link:
        lines += ["", f"Full vlog: {link}"]
    if credit:
        lines += ["", credit]
    tags = " ".join(t if t.startswith("#") else "#" + t for t in hashtags if t and t not in title)
    if tags:
        lines += ["", tags]
    return "\n".join(lines)[:2200]


def post_reel(cfg, path: Path, description: str, publish_at_iso: str | None, log=print) -> str:
    """Upload one Reel to the Page; scheduled for publish_at_iso (UTC 'YYYY-MM-DDTHH:MM:SSZ') or published now.
    Returns the Facebook video id."""
    import requests
    page, tok, v = _env("FB_PAGE_ID"), _env("FB_PAGE_TOKEN"), _ver(cfg)
    url = f"{GRAPH}/{v}/{page}/video_reels"
    st = _check(requests.post(url, data={"upload_phase": "start", "access_token": tok}, timeout=60))
    vid, up = st["video_id"], st.get("upload_url") or f"https://rupload.facebook.com/video-upload/{v}/{st['video_id']}"
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        _check(requests.post(up, data=f, timeout=900, headers={"Authorization": f"OAuth {tok}", "offset": "0",
                                                               "file_size": str(size)}))
    fin = {"upload_phase": "finish", "video_id": vid, "description": description, "access_token": tok,
           "video_state": "PUBLISHED"}
    if publish_at_iso:
        when = dt.datetime.strptime(publish_at_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        if when > dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15):     # Meta: >= 10 min ahead
            fin.update(video_state="SCHEDULED", scheduled_publish_time=str(int(when.timestamp())))
    _check(requests.post(url, data=fin, timeout=120))
    log(f"    facebook reel {vid} ({fin['video_state'].lower()})")
    return vid


def make_page_token(cfg=None) -> int:
    """FB_USER_TOKEN (short-lived, from Graph API Explorer) -> long-lived user token -> the Page's token, which
    never expires. Writes FB_PAGE_ID + FB_PAGE_TOKEN to .env. Nothing secret is printed."""
    import requests
    from .youtube import _write_env
    v = _ver(cfg)
    app_id, secret, short = _env("FB_APP_ID"), _env("FB_APP_SECRET"), _env("FB_USER_TOKEN")
    miss = [k for k, x in (("FB_APP_ID", app_id), ("FB_APP_SECRET", secret), ("FB_USER_TOKEN", short)) if not x]
    if miss:
        raise SystemExit(f"add {', '.join(miss)} to .env first (see joshiclip/facebook.py)")
    long_user = _check(requests.get(f"{GRAPH}/{v}/oauth/access_token", timeout=60, params={
        "grant_type": "fb_exchange_token", "client_id": app_id, "client_secret": secret,
        "fb_exchange_token": short}))["access_token"]
    pages = _check(requests.get(f"{GRAPH}/{v}/me/accounts", timeout=60,
                                params={"access_token": long_user, "fields": "id,name,access_token,tasks"})).get("data", [])
    if not pages:      # Page inside a Business portfolio: me/accounts is empty, the business lists it
        found = {}
        biz = _check(requests.get(f"{GRAPH}/{v}/me/businesses", timeout=60,
                                  params={"access_token": long_user, "fields": "id,name"})).get("data", [])
        for b in biz:
            for edge in ("owned_pages", "client_pages"):
                try:
                    for p in _check(requests.get(f"{GRAPH}/{v}/{b['id']}/{edge}", timeout=60, params={
                            "access_token": long_user, "fields": "id,name"})).get("data", []):
                        found[p["id"]] = p
                except RuntimeError:
                    pass
        for pid, p in found.items():
            try:
                t = _check(requests.get(f"{GRAPH}/{v}/{pid}", timeout=60,
                                        params={"access_token": long_user, "fields": "id,name,access_token"}))
                if t.get("access_token"):
                    pages.append({"id": pid, "name": t.get("name", p.get("name")), "access_token": t["access_token"],
                                  "tasks": ["CREATE_CONTENT"]})
            except RuntimeError:
                pass
        if not pages:
            raise SystemExit(f"no Page found: me/accounts is empty and {len(biz)} business portfolio(s) gave "
                             f"{len(found)} Page(s) without a token. In Graph API Explorer pick 'Get User Access "
                             "Token' and select your Page in the pop-up (Edit previous settings), or give this "
                             "Facebook account full control of the Page.")
    want = _env("FB_PAGE_NAME").lower()
    if len(pages) > 1 and not want:
        names = ", ".join(p["name"] for p in pages)
        raise SystemExit(f"you manage several Pages ({names}): put FB_PAGE_NAME=<exact name> in .env and run again")
    pg = next((p for p in pages if p["name"].lower() == want), None) if want else pages[0]
    if pg is None:
        raise SystemExit(f"no Page named {want!r}; you manage: " + ", ".join(p["name"] for p in pages))
    dbg = _check(requests.get(f"{GRAPH}/{v}/debug_token", timeout=60, params={
        "input_token": pg["access_token"], "access_token": f"{app_id}|{secret}"})).get("data", {})
    _write_env({"FB_PAGE_ID": pg["id"], "FB_PAGE_TOKEN": pg["access_token"]})
    exp = dbg.get("expires_at", 0)
    scopes = dbg.get("scopes") or []
    print(f"Page: {pg['name']} (id {pg['id']})  saved FB_PAGE_ID + FB_PAGE_TOKEN to .env")
    print("token expires: " + ("never" if not exp else dt.datetime.fromtimestamp(exp).isoformat()))
    need = {"pages_manage_posts", "pages_read_engagement", "pages_show_list"}
    if not need <= set(scopes):
        print("WARNING: missing permissions: " + ", ".join(sorted(need - set(scopes))))
    if "CREATE_CONTENT" not in (pg.get("tasks") or []):
        print("WARNING: your role on this Page can't create content")
    print("Now add FB_PAGE_ID and FB_PAGE_TOKEN as GitHub secrets in BOTH repos (s-clips and sorav-hood).")
    return 0
