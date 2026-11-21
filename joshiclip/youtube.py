"""YouTube Data API v3 upload (OAuth refresh token per channel) + one-time `joshiclip auth` to get that token.

Secrets per channel (env / GitHub Actions secrets), with <P> = the channel's `env` prefix in config daily.channels:
  <P>_CLIENT_ID, <P>_CLIENT_SECRET, <P>_REFRESH_TOKEN
A plain "API key" cannot upload: uploading always needs OAuth for the channel's Google account.
Quota: videos.insert costs 1600 units; a Google Cloud project gets 10,000 units/day by default -> 6 uploads/day.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


class QuotaExceeded(RuntimeError):
    pass


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
SHARED_CLIENT_JSON = "YOUTUBE_OAUTH_CLIENT_JSON"  # one OAuth client JSON for every channel (or <P>_CLIENT_JSON per channel)


def read_env_file(path: Path = ENV_FILE) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env() -> None:
    """Local runs: fill os.environ from .env (real env vars / GitHub secrets win)."""
    for k, v in read_env_file().items():
        os.environ.setdefault(k, v)


def _client_from_json(prefix: str) -> tuple[str | None, str | None]:
    import json
    raw = os.environ.get(f"{prefix}_CLIENT_JSON") or os.environ.get(SHARED_CLIENT_JSON)
    if not raw:
        return None, None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    info = info.get("installed") or info.get("web") or info
    return info.get("client_id"), info.get("client_secret")


def creds(prefix: str) -> dict:
    load_env()
    cid, csec = _client_from_json(prefix)
    return {"client_id": os.environ.get(f"{prefix}_CLIENT_ID") or cid,
            "client_secret": os.environ.get(f"{prefix}_CLIENT_SECRET") or csec,
            "refresh_token": os.environ.get(f"{prefix}_REFRESH_TOKEN")}


def have_secrets(prefix: str) -> bool:
    return all(creds(prefix).values())


def service(prefix: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    c = creds(prefix)
    for k, v in (("CLIENT_ID", c["client_id"]), ("CLIENT_SECRET", c["client_secret"]), ("REFRESH_TOKEN", c["refresh_token"])):
        if v:
            os.environ.setdefault(f"{prefix}_{k}", v)
    missing = [f"{prefix}_{k}" for k in ("CLIENT_ID", "CLIENT_SECRET", "REFRESH_TOKEN") if not os.environ.get(f"{prefix}_{k}")]
    if missing:
        raise RuntimeError(f"missing secrets: {', '.join(missing)}")
    cr = Credentials(None, refresh_token=os.environ[f"{prefix}_REFRESH_TOKEN"], client_id=os.environ[f"{prefix}_CLIENT_ID"],
                     client_secret=os.environ[f"{prefix}_CLIENT_SECRET"], token_uri=TOKEN_URI, scopes=SCOPES)
    return build("youtube", "v3", credentials=cr, cache_discovery=False)


def channel_name(svc) -> str:
    r = svc.channels().list(part="snippet", mine=True).execute()
    items = r.get("items") or []
    return items[0]["snippet"]["title"] if items else "?"


def _reason(e) -> str:
    try:
        import json
        return json.loads(e.content.decode())["error"]["errors"][0]["reason"]
    except Exception:  # noqa: BLE001
        return ""


def _fit_tags(tags: list[str], limit: int = 480) -> list[str]:
    """YouTube's hidden tags: 500 characters in total (a tag with a space counts 2 more for its quotes)."""
    out, n = [], 0
    for t in tags:
        t = t.lstrip("#").strip()
        if not t or t in out:
            continue
        c = len(t) + (2 if " " in t else 0) + (1 if out else 0)
        if n + c > limit:
            break
        out.append(t); n += c
    return out


def upload(svc, path: Path, title: str, description: str, tags: list[str], category_id: str = "24",
           publish_at: str | None = None, privacy: str = "public", log=print) -> str:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    status = {"privacyStatus": "private" if publish_at else privacy, "selfDeclaredMadeForKids": False,
              "embeddable": True}
    if publish_at:
        status["publishAt"] = publish_at
    title, description = (x.replace("<", "").replace(">", "") for x in (title, description))   # YouTube rejects < >
    body = {"snippet": {"title": title[:100], "description": description[:4900], "tags": _fit_tags(tags),
                        "categoryId": category_id, "defaultLanguage": "hi", "defaultAudioLanguage": "hi"},
            "status": status}
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = svc.videos().insert(part="snippet,status", body=body, media_body=media)
    resp, tries = None, 0
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            r = _reason(e)
            if e.resp.status == 403 and r in ("quotaExceeded", "uploadLimitExceeded", "rateLimitExceeded", "dailyLimitExceeded"):
                raise QuotaExceeded(r) from e
            if e.resp.status in (500, 502, 503, 504) and tries < 8:
                tries += 1; time.sleep(2 ** tries); continue
            raise
        except (ConnectionError, TimeoutError, OSError):
            if tries >= 8:
                raise
            tries += 1; time.sleep(2 ** tries)
    return resp["id"]


def check_token(prefix: str) -> tuple[bool, str]:
    """Exchange the refresh token for an access token -> (ok, message). Fails when Google put an expiry on the token:
    that means the OAuth consent screen is still in "Testing", where tokens die after 7 days."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request
    c = creds(prefix)
    if not all(c.values()):
        return False, f"{prefix}: missing " + ", ".join(k for k, v in c.items() if not v)
    data = urllib.parse.urlencode({**c, "grant_type": "refresh_token"}).encode()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(TOKEN_URI, data), timeout=30))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        hint = (" -> token expired or revoked. Publish the OAuth app (consent screen -> 'In production'), then run: "
                f"python -m joshiclip auth --prefix {prefix}") if "invalid_grant" in body else ""
        return False, f"{prefix}: token refresh failed ({e.code} {' '.join(body.split())[:160]}){hint}"
    req = urllib.request.Request("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                                 headers={"Authorization": f"Bearer {r['access_token']}"})
    items = json.load(urllib.request.urlopen(req, timeout=30)).get("items") or []
    name = items[0]["snippet"]["title"] if items else "? (this Google account has no channel)"
    msg = f"{prefix}: uploads go to channel '{name}'"
    if r.get("refresh_token_expires_in"):
        days = int(r["refresh_token_expires_in"]) / 86400
        return False, (f"{msg}, BUT this token expires in {days:.1f} days: the OAuth consent screen is in 'Testing'. "
                       f"Publish it ('In production') and run `python -m joshiclip auth --prefix {prefix}` again.")
    return True, msg


def check_all(cfg, only: str | None = None) -> int:
    bad = 0
    for ch in cfg["daily"]["channels"]:
        if only and ch["name"] != only:
            continue
        ok, msg = check_token(ch["env"])
        print(("  OK   " if ok else "  FAIL ") + f"[{ch['name']}] {msg}")
        bad += not ok
    return 1 if bad else 0


def _write_env(updates: dict, path: Path = ENV_FILE) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        k = line.split("=", 1)[0].strip()
        if k in updates:
            lines[i] = f"{k}={updates[k]}"
            seen.add(k)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def auth(client_secret_json: str | None = None, prefix: str = "YT_A") -> int:
    """Run ONCE per channel on your own PC: browser login with the channel's Google account. The 3 secrets are
    written into .env (git-ignored) and printed for GitHub. Client: the given JSON file, else <P>_CLIENT_JSON /
    YOUTUBE_OAUTH_CLIENT_JSON from .env."""
    import json
    from google_auth_oauthlib.flow import InstalledAppFlow
    load_env()
    if client_secret_json:
        info = json.loads(Path(client_secret_json).read_text())
    else:
        raw = os.environ.get(f"{prefix}_CLIENT_JSON") or os.environ.get(SHARED_CLIENT_JSON)
        if not raw:
            raise SystemExit(f"no OAuth client: pass the client_secret.json path or set {SHARED_CLIENT_JSON} in .env")
        info = json.loads(raw)
    flow = InstalledAppFlow.from_client_config(info, SCOPES)
    cr = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    from googleapiclient.discovery import build
    name = channel_name(build("youtube", "v3", credentials=cr, cache_discovery=False))
    c = info.get("installed") or info.get("web") or {}
    vals = {f"{prefix}_CLIENT_ID": c.get("client_id"), f"{prefix}_CLIENT_SECRET": c.get("client_secret"),
            f"{prefix}_REFRESH_TOKEN": cr.refresh_token}
    _write_env(vals)
    os.environ.update(vals)
    print(f"\nLogged in as channel: {name}   (saved to .env)")
    print("Add these 3 as GitHub repository secrets (Settings -> Secrets and variables -> Actions):\n")
    for k, v in vals.items():
        print(f"  {k:22s} = {v}")
    ok, msg = check_token(prefix)
    print(("\nCHECK OK: " if ok else "\nCHECK FAILED: ") + msg)
    print("Keep these private. Never commit them.")
    return 0 if ok else 1
