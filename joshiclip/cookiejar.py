"""Keep the YouTube cookies alive on GitHub runners, and warn before they die.

yt-dlp refreshes the session cookies every time it uses them and writes them back to the --cookies file. On a
GitHub runner that file was thrown away after each run, so every run started again from the (aging) secret.

  load   -> cookies for this run: the refreshed jar from the previous run (Actions cache) if it was made from the
            CURRENT YTDLP_COOKIES secret, else the secret itself (a refreshed secret always wins)
  save   -> the jar after the run, encrypted, into the cache directory (key derived from the secret: a runner
            without the secret - e.g. a fork - can't read it)
  check  -> 10-second test that YouTube accepts the cookies + how many days the login cookies have left;
            exit 2 = rejected, exit 3 = expiring soon (the workflow opens a GitHub issue -> email to the owner)

Encryption: SHAKE-256 keystream (stdlib only) keyed by SHA-256(secret), random 16-byte nonce, HMAC-SHA256 tag.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
import time
from pathlib import Path

LOGIN_COOKIES = ("__Secure-1PSID", "__Secure-3PSID", "SID", "SAPISID", "LOGIN_INFO")


def _key(secret: str) -> bytes:
    return hashlib.sha256(b"joshiclip-cookies-v1\0" + secret.encode("utf-8")).digest()


def _stream(key: bytes, nonce: bytes, n: int) -> bytes:
    return hashlib.shake_256(key + nonce).digest(n)


def encrypt(data: bytes, secret: str) -> bytes:
    k, nonce = _key(secret), os.urandom(16)
    ct = bytes(a ^ b for a, b in zip(data, _stream(k, nonce, len(data))))
    return b"JCK1" + nonce + hmac.new(k, nonce + ct, hashlib.sha256).digest() + ct


def decrypt(blob: bytes, secret: str) -> bytes | None:
    """None when the blob was made from another secret (or is damaged)."""
    if len(blob) < 52 or blob[:4] != b"JCK1":
        return None
    k, nonce, tag, ct = _key(secret), blob[4:20], blob[20:52], blob[52:]
    if not hmac.compare_digest(tag, hmac.new(k, nonce + ct, hashlib.sha256).digest()):
        return None
    return bytes(a ^ b for a, b in zip(ct, _stream(k, nonce, len(ct))))


def load(secret: str, cache_dir: Path, out: Path) -> str:
    """Write this run's cookies to `out`. Returns 'refreshed' (cached jar) or 'secret'."""
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = cache_dir / "jar.bin"
    if blob.exists():
        txt = decrypt(blob.read_bytes(), secret)
        if txt and b"youtube" in txt:
            out.write_bytes(txt)
            return "refreshed"
    out.write_text(secret, encoding="utf-8")
    return "secret"


def save(secret: str, jar: Path, cache_dir: Path) -> bool:
    if not jar.exists() or jar.stat().st_size < 100:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "jar.bin").write_bytes(encrypt(jar.read_bytes(), secret))
    return True


def days_left(jar_text: str, now: float | None = None) -> float | None:
    """Days until the first login cookie expires (None: no login cookie with an expiry date)."""
    now = time.time() if now is None else now
    exp = []
    for line in jar_text.splitlines():
        line = line[len("#HttpOnly_"):] if line.startswith("#HttpOnly_") else line
        f = line.split("\t")
        if line.startswith("#") or len(f) < 7:
            continue
        if f[5] in LOGIN_COOKIES and f[4].isdigit() and int(f[4]) > 0:
            exp.append(int(f[4]))
    return round((min(exp) - now) / 86400, 1) if exp else None


def check(jar: Path, video: str, warn_days: float = 14.0) -> int:
    """0 ok, 2 YouTube rejects the cookies, 3 login cookies expire within warn_days."""
    if not jar.exists():
        print("no cookies file")
        return 2
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "--cookies", str(jar), "--simulate", "--no-warnings",
                        "-f", "bv*[height<=1080]+ba/b", "--print", "%(id)s %(height)sp",
                        f"https://www.youtube.com/watch?v={video}"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print("YouTube REJECTED the cookies: " + out[-300:])
        return 2
    left = days_left(jar.read_text(encoding="utf-8", errors="ignore"))
    print(f"cookies OK ({out.splitlines()[-1] if out else ''}); login cookies expire in "
          f"{'unknown' if left is None else f'{left} days'}")
    if left is not None and left < warn_days:
        return 3
    return 0


def main(argv: list[str]) -> int:
    """python -m joshiclip.cookiejar load|save|check  (secret from env YTDLP_COOKIES_SECRET)."""
    secret = os.environ.get("YTDLP_COOKIES_SECRET", "")
    cache = Path(os.environ.get("JOSHICLIP_COOKIE_CACHE", "ytjar"))
    jar = Path(os.environ.get("JOSHICLIP_YTDLP_COOKIES", "cookies.txt"))
    cmd = argv[0] if argv else ""
    if cmd == "load":
        if not secret.strip():
            print("no YTDLP_COOKIES secret")
            return 0
        print("cookies: " + load(secret, cache, jar))
        return 0
    if cmd == "save":
        print("cookies saved for next run" if secret.strip() and save(secret, jar, cache) else "cookies not saved")
        return 0
    if cmd == "check":
        return check(jar, argv[1] if len(argv) > 1 else "TeCrjPZ9fds")
    print(main.__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
