"""Record a YouTube video by PLAYING it in Chrome instead of downloading it (JOSHICLIP_FETCH=record).

GitHub's datacenter IPs get "Sign in to confirm you're not a bot" from yt-dlp, so on Actions nothing is downloaded:
the vlog is played in Google Chrome and recorded. Goals: 4K, no dropped / frozen frames, exact lip-sync.

PICTURE (frame-exact): every frame the player presents is taken with requestVideoFrameCallback, stamped with its
  own position in the vlog (metadata.mediaTime) and encoded at full resolution with WebCodecs (H.264, high bitrate).
  Every frame carries its true vlog time, so buffering, ads and pauses cannot shift, freeze or lag the picture.
  When the encoder falls behind, playback is PAUSED until it catches up (a frame is never skipped for speed), and
  4K plays at 0.5x (or slower) so the decoder never drops one either. Missing frames are counted; if any go missing
  the clip is recorded again, slower.
SOUND: a separate 1x pass (sound can't be slowed down). The recorder pauses the same millisecond an ad starts or the
  player buffers (media events + a MutationObserver on the ad class); a clip whose sound pass had ANY pause is
  recorded again, so no clip contains a gap or a jump. The recorder is armed while the player is parked at the
  start, so the sound's first sample is the start frame's sound: measured lip-sync error 0-1 ms vs the original.
CAPTIONS: the Hindi auto-captions come from the player's own caption request (the same json3 yt-dlp writes);
  captions are never part of the recorded frames.

Modes (env JOSHICLIP_FETCH or .env): ytdlp (default) | record | auto (yt-dlp, and record if it fails).
"""
from __future__ import annotations

import base64
import json
import os
import struct
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import media, tools

QUALITIES = ["hd2160", "hd1440", "hd1080", "hd720", "large", "medium", "small", "tiny"]   # YouTube's names
HEIGHT = {"hd2160": 2160, "hd1440": 1440, "hd1080": 1080, "hd720": 720, "large": 480, "medium": 360, "small": 240,
          "tiny": 144}
DEFAULTS = {"clip_max_quality": "hd2160", "clip_min_quality": "hd720", "clip_bitrate_mbps": 20,
            "analysis_quality": "medium", "analysis_bitrate_mbps": 2, "audio_kbps": 256, "audio_quality": "tiny",
            "max_missing_frac": 0.01, "max_gap_frames": 2, "min_rate": 0.25, "audio_retries": 3, "headless": False, "pad_wait_s": 2.0,
            "start_timeout_s": 45, "sound_offset_ms": 0.0, "max_repairs": 400,
            "no_frame_timeout_s": 90, "tail_s": 0.5, "codecs": ["avc1.640034", "avc1.640033", "avc1.640028", "vp09.00.51.08", "vp8"]}
FOURCC = {"avc1": b"H264", "vp09": b"VP90", "vp8": b"VP80", "av01": b"AV01"}


class Stuck(RuntimeError):
    """This quality never started playing (too slow to buffer here): try the next lower one."""


def mode() -> str:
    m = (os.environ.get("JOSHICLIP_FETCH") or tools._dotenv().get("JOSHICLIP_FETCH") or "ytdlp").strip().lower()
    if m not in ("ytdlp", "record", "auto"):
        raise RuntimeError(f"JOSHICLIP_FETCH must be ytdlp, record or auto (got {m!r})")
    return m


def _rc(cfg) -> dict:
    return dict(DEFAULTS, **(cfg.get("record") or {}))


def bitrate(r: dict, height: int) -> int:
    """bits/s for this height: clip_bitrate_mbps at 1080p, x (h/1080)^1.5 (4K ~ 57 Mbps at the default 20)."""
    return int(r["clip_bitrate_mbps"] * 1e6 * (max(height, 144) / 1080) ** 1.5)


def missing_frames(deltas, span: float | None = None) -> tuple[float, int, int]:
    """(fps, frames missing, longest run of missing frames) from consecutive recorded frames' vlog times.
    The frame interval is the SHORTEST common gap (10th percentile), not the median: when most frames are lost
    the median is itself a gap. With `span` (b - a) frames missing at the start / end count too."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[d > 1e-4]
    if not len(d):
        return 0.0, int(span * 30) if span else 0, int(span * 30) if span else 0
    fi = float(np.percentile(d, 10))
    m = np.maximum(0, np.round(d / fi) - 1)
    miss, gap = int(m.sum()), int(m.max())
    if span:
        expect = int(round(span / fi))
        tail = max(0, expect - (len(d) + 1) - miss)
        miss, gap = miss + tail, max(gap, tail)
    return 1.0 / fi, miss, gap


# ---------------------------------------------------------------- browser
def _cookies() -> list[dict]:
    """yt-dlp's Netscape cookies.txt (YTDLP_COOKIES secret) -> Playwright cookies: logged in = fewer checks/ads."""
    p = tools.cookies_path()
    out = []
    if not p:
        return out
    for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
        http_only = line.startswith("#HttpOnly_")
        if http_only:
            line = line[len("#HttpOnly_"):]
        f = line.split("\t")
        if line.startswith("#") or len(f) < 7:
            continue
        dom, _, path, secure, exp, name, value = f[:7]
        if "youtube" not in dom and "google" not in dom:
            continue
        c = {"name": name, "value": value.strip(), "domain": dom, "path": path or "/", "secure": secure.upper() == "TRUE",
             "httpOnly": http_only, "sameSite": "Lax"}
        if exp.isdigit() and int(exp) > 0:
            c["expires"] = int(exp)
        out.append(c)
    return out


class Browser:
    """One Chrome for a whole run; clips are recorded one after another."""

    def __init__(self, cfg, via_proxy: bool = True, log=print):
        from playwright.sync_api import sync_playwright
        self.r, self.log = _rc(cfg), log
        self.pw = sync_playwright().start()
        args = ["--autoplay-policy=no-user-gesture-required", "--disable-background-media-suspend",
                "--disable-renderer-backgrounding", "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows", "--disable-blink-features=AutomationControlled",
                # Windows stops painting a covered Chrome window -> no frames to capture. Keep painting always.
                "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling"]
        kw = {"headless": bool(self.r["headless"]), "args": args}
        proxy = tools.ytdlp_proxy()
        if proxy and via_proxy:
            kw["proxy"] = {"server": proxy}
        try:
            self.browser = self.pw.chromium.launch(channel="chrome", **kw)    # real Chrome: all codecs, normal fingerprint
        except Exception:  # noqa: BLE001
            log("    Google Chrome not found; using Playwright's Chromium")
            self.browser = self.pw.chromium.launch(**kw)
        self.ctx = self.browser.new_context(viewport={"width": 1920, "height": 1080}, locale="en-US",
                                            device_scale_factor=1)
        ck = _cookies()
        if ck:
            try:
                self.ctx.add_cookies(ck)
            except Exception as e:  # noqa: BLE001
                log(f"    cookies not loaded into the browser ({str(e)[:80]})")
        self._vf = self._af = None
        self._a_us = self._vn = 0
        self._done = False
        self._caps: dict[str, bytes] = {}
        self.ctx.expose_binding("__jcChunk", self._chunk)
        self.ctx.expose_binding("__jcDone", lambda _src: setattr(self, "_done", True))
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(60_000)
        self.page.on("response", self._on_response)
        self.vid = None

    def close(self):
        for f in (self.ctx.close, self.browser.close, self.pw.stop):
            try:
                f()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------ page
    def _on_response(self, resp):
        if "/api/timedtext" not in resp.url:
            return
        q = parse_qs(urlparse(resp.url).query)
        if q.get("tlang") or q.get("fmt", ["json3"])[0] != "json3":
            return
        try:
            body = resp.body()
            if b'"events"' in body:
                self._caps[(q.get("lang") or ["?"])[0] + ("-asr" if q.get("kind") == ["asr"] else "")] = body
        except Exception:  # noqa: BLE001
            pass

    def open(self, vid: str):
        if self.vid == vid:
            return
        p = self.page
        p.goto(f"https://www.youtube.com/watch?v={vid}", wait_until="domcontentloaded")
        for sel in ('button:has-text("Accept all")', 'button:has-text("Reject all")'):   # EU consent wall
            try:
                p.locator(sel).first.click(timeout=2500)
                break
            except Exception:  # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < 60:
            st = p.evaluate("""() => {
                const P = document.getElementById('movie_player'), t = document.body.innerText || '';
                if (/confirm you.re not a bot|unusual traffic/i.test(t)) return 'bot';
                const e = document.querySelector('.ytp-error-content-wrap-reason, .ytp-error');
                if (e && e.offsetParent && e.innerText.trim()) return 'error:' + e.innerText.trim().slice(0, 200);
                return P && P.getDuration && P.getDuration() > 0 && document.querySelector('video') ? 'ok' : '';
            }""")
            if st == "ok":
                break
            if st == "bot":
                raise RuntimeError("YouTube: Sign in to confirm you're not a bot (browser)")
            if st.startswith("error:"):
                raise RuntimeError(f"YouTube player error: {st[6:]}")
            p.wait_for_timeout(1000)
        else:
            raise RuntimeError("YouTube player did not load in 60 s")
        p.evaluate("""() => { const P = document.getElementById('movie_player');
            try { P.setAutonav && P.setAutonav(false); } catch (e) {} P.unMute && P.unMute(); P.setVolume && P.setVolume(100); }""")
        self.vid = vid

    def info(self) -> dict:
        return self.page.evaluate("""() => { const P = document.getElementById('movie_player'), d = P.getVideoData();
            return {id: d.video_id, title: d.title, author: d.author, duration: P.getDuration(),
                    qualities: P.getAvailableQualityLevels()}; }""")

    def _skip_ads(self, limit_s: float = 240):
        """Pre-roll ads: wait them out (clicking Skip when it appears)."""
        t0 = time.time()
        while time.time() - t0 < limit_s:
            ad = self.page.evaluate("""() => { const P = document.getElementById('movie_player');
                if (!P.classList.contains('ad-showing')) return false;
                const b = document.querySelector('.ytp-skip-ad-button, .ytp-ad-skip-button, .ytp-ad-skip-button-modern');
                if (b) b.click(); const v = document.querySelector('video'); if (v && v.paused) v.play(); return true; }""")
            if not ad:
                return
            self.page.wait_for_timeout(300)
        raise RuntimeError("an ad kept playing for 4 min")

    def captions(self, out_dir: Path, vid: str, log=print) -> Path | None:
        """Turn on the Hindi track so the player fetches it, save the json3 it receives, turn captions off again."""
        try:
            self.page.evaluate("""() => { const P = document.getElementById('movie_player');
                P.loadModule && P.loadModule('captions');
                const L = (P.getOption('captions', 'tracklist') || []).filter(t => (t.languageCode || '').startsWith('hi'));
                const t = L.find(t => t.kind === 'asr') || L[0]; if (t) P.setOption('captions', 'track', t); }""")
            for _ in range(20):
                if self._caps:
                    break
                self.page.wait_for_timeout(500)
            self.page.evaluate("() => { const P = document.getElementById('movie_player'); P.unloadModule && P.unloadModule('captions'); }")
        except Exception as e:  # noqa: BLE001
            log(f"    captions: {str(e)[:80]}")
        key = next((k for k in ("hi-asr", "hi") if k in self._caps), None) or next(
            (k for k in self._caps if k.startswith("hi")), None)
        if not key:
            log("    no Hindi captions from the player (Whisper will transcribe)")
            return None
        p = out_dir / f"{vid}.{'hi-orig' if key.endswith('-asr') else 'hi'}.json3"
        p.write_bytes(self._caps[key])
        log(f"    captions captured: {p.name}")
        return p

    # ------------------------------------------------------------ data from the page
    def _chunk(self, _src, kind, b64, ts_us, _key):
        data = base64.b64decode(b64)
        if kind == "v":
            if self._v0 is None:               # the first frame is t=0 of the file (the sound is trimmed to match)
                self._v0 = int(ts_us)
            self._vf.write(struct.pack("<IQ", len(data), max(0, int(ts_us) - self._v0)))
            self._vf.write(data)
            self._vn += 1
        else:
            self._af.write(data)

    # ------------------------------------------------------------ one pass
    def _prep_youtube(self, vid: str, a: float, quality: str, rate: float) -> str:
        """Quality + speed set, then parked PAUSED exactly at `a` with that quality buffered. Returns the quality."""
        self.open(vid)
        self._skip_ads()
        p = self.page
        q = p.evaluate("""(want) => { const P = document.getElementById('movie_player'), ALL = %s;
            const av = P.getAvailableQualityLevels(); const ok = ALL.slice(ALL.indexOf(want)).find(x => av.includes(x)) || 'auto';
            P.setPlaybackQualityRange && P.setPlaybackQualityRange(ok, ok); P.setPlaybackQuality && P.setPlaybackQuality(ok);
            return ok; }""" % json.dumps(QUALITIES), quality)
        p.evaluate("(r) => document.getElementById('movie_player').setPlaybackRate(r)", rate)
        pre = max(0.0, a - float(self.r["pad_wait_s"]))
        p.evaluate("(t) => { const P = document.getElementById('movie_player'); P.seekTo(t, true); P.playVideo(); }", pre)
        want_h, lim, t0 = HEIGHT.get(q, 0), float(self.r["start_timeout_s"]), time.time()
        while True:                           # the requested resolution actually playing
            h = p.evaluate("() => { const v = document.querySelector('video'); return v.readyState >= 3 && !v.paused ? v.videoHeight : 0; }")
            if h and (h >= want_h * 0.95 or time.time() - t0 > lim / 2):
                break
            if time.time() - t0 > lim:
                raise Stuck(f"{q} did not start playing in {lim:.0f} s")
            p.wait_for_timeout(250)
        self._skip_ads()
        p.evaluate("(t) => { const P = document.getElementById('movie_player'); P.pauseVideo(); P.seekTo(t, true); }", a)
        self._wait_parked(p, a)
        return q

    def _wait_parked(self, p, a: float):
        t0 = time.time()
        while not p.evaluate("(a) => { const v = document.querySelector('video'); return !v.seeking && v.readyState >= 3 && Math.abs(v.currentTime - a) < 0.3; }", a):
            if time.time() - t0 > float(self.r["start_timeout_s"]):
                raise Stuck(f"seek to {a:.1f}s did not finish")
            p.wait_for_timeout(50)

    def _pass(self, p, a: float, b: float, rate: float, out_v: Path | None, out_a: Path | None, vbps: int = 0,
              log=print, label="") -> dict:
        """Run _JS on the parked page: picture -> out_v (IVF, vlog-time stamps relative to a), sound -> out_a (webm)."""
        o = {"a": a, "b": b, "video": out_v is not None, "audio": out_a is not None, "vbps": vbps,
             "abps": int(self.r["audio_kbps"] * 1000), "codecs": list(self.r["codecs"]), "gop": 30,
             "max_repairs": int(self.r["max_repairs"])}
        init = p.evaluate(_JS, o)
        if init.get("err"):
            raise RuntimeError(f"recorder: {init['err']}")
        self._done, self._vn, self._a_us, self._v0 = False, 0, int(round(a * 1e6)), None
        self._vf = open(out_v, "wb") if out_v else None
        self._af = open(out_a, "wb") if out_a else None
        try:
            if self._vf:
                self._vf.write(b"DKIF" + struct.pack("<HH4sHHIII", 0, 32, FOURCC[init["codec"].split(".")[0]],
                                                     init["w"], init["h"], 1_000_000, 1, 0) + bytes(4))
            p.evaluate("() => window.__jcGo()")
            limit = (b - a) / rate * (3.5 if out_v else 2.0) + 180
            t0 = last_log = t_new = time.time()
            last_log, seen = 0.0, -1
            while not self._done:
                p.wait_for_timeout(200)
                st = p.evaluate("() => { const S = window.__jc; return {t: S.t, frames: S.frames, seams: S.seams, "
                                "ad_s: S.ad_s, throttles: S.throttles, repairs: S.repairs, err: S.err, bot: S.bot}; }")
                if st.get("err"):
                    raise RuntimeError(f"recording failed: {st['err']}")
                if st.get("bot"):
                    raise RuntimeError("YouTube: Sign in to confirm you're not a bot (browser, during playback)")
                el = time.time() - t0
                if out_v and st.get("frames") != seen:
                    seen, t_new = st.get("frames"), time.time()
                if el > limit or (out_v and time.time() - t_new > float(self.r["no_frame_timeout_s"])):
                    raise Stuck(f"{label} {a:.0f}-{b:.0f}s too slow / stuck at {st.get('t') or 0:.0f}s "
                                f"({st.get('frames')} frames, {st.get('repairs', 0)} repairs, {el:.0f} s)")
                if el - last_log > 60:
                    last_log = el
                    log(f"    {label} {(st.get('t') or a) - a:.0f}/{b - a:.0f} s  frames {st.get('frames')}  "
                        f"encoder waits {st.get('throttles')}  sound pauses {st.get('seams')}  ads {st.get('ad_s') or 0:.0f}s")
            st = p.evaluate("() => { const S = Object.assign({}, window.__jc); delete S.deltas; return S; }")
            deltas = p.evaluate("() => window.__jc.deltas")
        finally:
            try:
                p.evaluate("() => { window.__jcGen = (window.__jcGen || 0) + 1; }")   # stop whatever still runs
            except Exception:  # noqa: BLE001
                pass
            for f in (self._vf, self._af):
                if f:
                    f.close()
            self._vf = self._af = None
        st["codec"] = init.get("codec")
        if out_v:
            with open(out_v, "r+b") as f:     # frame count into the IVF header
                f.seek(24)
                f.write(struct.pack("<I", self._vn))
            st["fps"], st["missing"], st["max_gap"] = missing_frames(deltas, b - a if b - a < 3600 else None)
            st["missing_frac"] = st["missing"] / max(1, st["missing"] + self._vn)
            st["lead_s"] = ((self._v0 or self._a_us) - self._a_us) / 1e6   # first frame's time after `a`
        return st

    # ------------------------------------------------------------ public
    def mux(self, v: Path, a: Path, out: Path, lead: float = 0.0):
        """IVF picture (exact vlog-time stamps) + sound (starts at `a`: recorder armed while parked) -> mkv (FLAC)."""
        # the sound starts at `a`, the picture's first frame at a + lead: cut `lead` off the sound -> both start at 0
        off = lead + float(self.r["sound_offset_ms"]) / 1000
        sh = (f",atrim=start={off:.4f},asetpts=PTS-STARTPTS" if off > 0.0005 else
              f",adelay={-off * 1000:.2f}:all=1" if off < -0.0005 else "")
        part = out.with_name(out.stem + ".part.mkv")
        media.run([media.ff(), "-v", "error", "-y", "-copyts", "-i", str(v), "-i", str(a), "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-af", f"asetpts=N/SR/TB,aresample=48000{sh}", "-c:a", "flac", str(part)])
        os.replace(part, out)

    def record_video(self, vid, a, b, out_v: Path, quality: str, rate: float, log=print) -> dict:
        q = self._prep_youtube(vid, a, quality, rate)
        st = self._pass(self.page, a, b, rate, out_v, None, bitrate(self.r, HEIGHT.get(q, 1080)), log, "picture")
        st["quality"], st["rate"] = q, rate
        return st

    def record_audio(self, vid, a, b, out_a: Path, log=print) -> dict:
        self._prep_youtube(vid, a, self.r["audio_quality"], 1.0)
        return self._pass(self.page, a, b, 1.0, None, out_a, log=log, label="sound")


# Runs in the page (YouTube, or any page with one <video>). Sets everything up and returns {codec, w, h};
# window.__jcGo() then starts playback.
_JS = r"""
async (o) => {
  const P = document.getElementById('movie_player');            // null on a plain (non-YouTube) page
  const v = P ? (document.querySelector('video.html5-main-video') || document.querySelector('video')) : document.querySelector('video');
  const gen = window.__jcGen = (window.__jcGen || 0) + 1;       // a newer take (or an abort) stops this one
  const alive = () => gen === window.__jcGen;
  const S = window.__jc = {frames: 0, seams: 0, ad_s: 0, throttles: 0, repairs: 0, t: v.currentTime, err: null, done: false,
                           vdone: !o.video, adone: !o.audio, deltas: [], w: v.videoWidth, h: v.videoHeight};
  const isAd = () => !!(P && P.classList.contains('ad-showing'));
  const play = () => { if (P) P.playVideo(); else v.play(); };
  const pause = () => { if (P) P.pauseVideo(); else v.pause(); };
  const b64 = (u) => { let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
  let vchain = Promise.resolve(), achain = Promise.resolve();
  const finish = () => { if (S.vdone && S.adone && !S.done) { S.done = true; pause();
    Promise.all([vchain, achain]).then(() => window.__jcDone()); } };
  let tAd = null;
  const watch = setInterval(() => {                               // ads: count + skip; progress
    S.t = v.currentTime;
    const ad = isAd();
    if (ad) { const k = document.querySelector('.ytp-skip-ad-button, .ytp-ad-skip-button, .ytp-ad-skip-button-modern'); if (k) k.click(); }
    if (ad && tAd === null) tAd = performance.now();
    if (!ad && tAd !== null) { S.ad_s += (performance.now() - tAd) / 1000; tAd = null; }
    if (P && /confirm you.re not a bot/i.test(document.body.innerText || '')) S.bot = true;
    if (S.done || !alive()) clearInterval(watch);
  }, 20);

  // ---------------- picture: every presented frame, stamped with its vlog time, encoded with WebCodecs
  let codec = null;
  if (o.video) {
    const W = v.videoWidth, H = v.videoHeight;
    let cfg = null;
    for (const c of o.codecs) {
      const t = {codec: c, width: W, height: H, bitrate: o.vbps, bitrateMode: 'variable', latencyMode: 'quality', framerate: 60};
      if (c.startsWith('avc1')) t.avc = {format: 'annexb'};
      try { if ((await VideoEncoder.isConfigSupported(t)).supported) { cfg = t; break; } } catch (e) {}
    }
    if (!cfg) return {err: 'no WebCodecs encoder for ' + W + 'x' + H};
    codec = cfg.codec;
    let throttled = false, last = -1, n = 0, fi = 0, repairing = false, repAt = -1, repTries = 0;
    const enc = new VideoEncoder({
      output: (ch) => { const u = new Uint8Array(ch.byteLength); ch.copyTo(u); const ts = ch.timestamp, key = ch.type === 'key';
                        if (alive()) vchain = vchain.then(() => alive() && window.__jcChunk('v', b64(u), ts, key)); },
      error: (e) => { S.err = 'encoder: ' + e; }});
    enc.configure(cfg);
    const relax = () => { if (throttled && !repairing && enc.encodeQueueSize <= 1 && !S.vdone) { throttled = false; play(); } };
    // self-healing: a gap in the vlog times = the player skipped frames -> seek back to the last recorded frame and
    // take the skipped ones (frames already recorded are ignored, so the order stays exact)
    const repair = () => {
      repairing = true; S.repairs++; pause();
      let fin = false;
      const done = () => { if (fin) return; fin = true; v.removeEventListener('seeked', done);
        setTimeout(() => { repairing = false; if (!throttled) play(); }, 30); };
      v.addEventListener('seeked', done);
      setTimeout(done, 1500);          // the player may ignore a tiny seek and never fire 'seeked': go on anyway
      const t = last + Math.min(fi * 0.25, 0.004);
      v.currentTime = t;               // the element itself: exact, and always fires 'seeked'
    };
    enc.addEventListener('dequeue', relax);
    const relaxTimer = setInterval(() => { relax(); if (S.vdone) clearInterval(relaxTimer); }, 50);
    const endV = async () => { if (S.vdone) return; S.vdone = true; try { await enc.flush(); enc.close(); } catch (e) {} finish(); };
    const onFrame = (now, md) => {
      if (S.vdone || !alive()) { try { enc.close(); } catch (e) {} return; }
      try {
        const mt = md.mediaTime;
        if (!isAd() && !repairing && mt > last + 0.0005 && mt >= o.a - 0.0005) {
          if (mt >= o.b) { endV(); return; }
          if (last >= 0 && n >= 3 && fi > 0 && mt - last > 1.5 * fi) {
            if (repAt === last) repTries++; else { repAt = last; repTries = 0; }
            if (repTries < 4 && S.repairs < o.max_repairs) { repair(); v.requestVideoFrameCallback(onFrame); return; }
          }
          if (v.videoWidth !== W || v.videoHeight !== H) throw new Error('resolution changed to ' + v.videoWidth + 'x' + v.videoHeight);
          if (last >= 0) { const dd = mt - last; S.deltas.push(dd); if (dd > 0.004 && (fi === 0 || dd < fi)) fi = dd; }
          const f = new VideoFrame(v, {timestamp: Math.round(mt * 1e6)});
          enc.encode(f, {keyFrame: n % o.gop === 0}); f.close();
          n++; last = mt; S.frames = n;
          if (enc.encodeQueueSize > 3 && !throttled) { throttled = true; S.throttles++; pause(); }   // never drop: wait
        }
      } catch (e) { S.err = String(e); S.vdone = true; finish(); return; }
      v.requestVideoFrameCallback(onFrame);
    };
    v.requestVideoFrameCallback(onFrame);
    v.addEventListener('ended', endV);
  }

  // ---------------- sound: 1x, recorder paused the instant an ad / buffering / seek happens
  if (o.audio) {
    let st = v.captureStream();
    for (let i = 0; i < 50 && !st.getAudioTracks().length; i++) { await new Promise(r => setTimeout(r, 100)); st = v.captureStream(); }
    if (!st.getAudioTracks().length) return {err: 'no audio track in the captured stream'};
    const rec = new MediaRecorder(new MediaStream(st.getAudioTracks()), {mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: o.abps});
    rec.ondataavailable = (e) => { if (e.data && e.data.size && alive()) { const blob = e.data;
      achain = achain.then(async () => window.__jcChunk('a', b64(new Uint8Array(await blob.arrayBuffer())), 0, false)); } };
    let started = false, stopped = false;
    const bad = () => isAd() || v.paused || v.seeking || v.readyState < 3;
    const upd = () => {
      if (stopped || rec.state === 'inactive') return;
      if (bad()) { if (started && rec.state === 'recording') { rec.pause(); S.seams++; } }
      else { started = true; if (rec.state === 'paused') rec.resume(); }
    };
    for (const ev of ['waiting', 'pause', 'seeking', 'stalled', 'emptied', 'playing', 'play', 'seeked', 'canplay']) v.addEventListener(ev, upd);
    if (P) new MutationObserver(upd).observe(P, {attributes: true, attributeFilter: ['class']});
    const stopA = () => { if (stopped) return; stopped = true;
      rec.onstop = () => { achain.then(() => { S.adone = true; finish(); }); }; rec.stop(); };
    const iv = setInterval(() => { if (!alive()) { clearInterval(iv); try { rec.stop(); } catch (e) {} return; }
      if (!isAd() && (v.currentTime >= o.b + 0.3 || v.ended)) { clearInterval(iv); stopA(); } else upd(); }, 5);
    rec.start(1000);                                                // armed while parked: no sound flows until play
  }
  window.__jcGo = () => play();
  return {codec: codec, w: v.videoWidth, h: v.videoHeight};
}
"""


# ---------------------------------------------------------------- pipeline entry points
def _with_routes(cfg, fn, log=print):
    """Proxy first (if set), then direct; same idea as fast._ytdlp."""
    routes = [True, False] if tools.ytdlp_proxy() else [True]
    last = None
    for via in routes:
        try:
            with Browser(cfg, via, log) as br:
                return fn(br)
        except RuntimeError as e:
            if "not a bot" not in str(e):
                raise
            last = e
            log(f"    browser bot-checked {'via proxy' if via and len(routes) > 1 else 'direct'}")
    raise last


def record_full(cfg, vid: str, wd: Path, log=print) -> tuple[Path, dict]:
    """Whole vlog at analysis quality (360p, picture + sound in one 1x pass) + Hindi captions + info json.
    Takes as long as the vlog. A sound pause here only shifts the analysis by ms; every clip is aligned again."""
    r = _rc(cfg)
    out = wd / "src360.mkv"

    def go(br):
        br.open(vid)
        inf = br.info()
        log(f"  recording the whole vlog in Chrome ({inf['duration'] / 60:.1f} min at {r['analysis_quality']}) - no download")
        br._skip_ads()
        br.captions(wd, vid, log)
        q = br._prep_youtube(vid, 0.0, r["analysis_quality"], 1.0)
        tv, ta = wd / "rec_full.ivf", wd / "rec_full.webm"
        st = br._pass(br.page, 0.0, inf["duration"] + 1.0, 1.0, tv, ta, int(r["analysis_bitrate_mbps"] * 1e6), log,
                      "full vlog")
        br.mux(tv, ta, out, st.get("lead_s", 0.0))
        tv.unlink(missing_ok=True)
        ta.unlink(missing_ok=True)
        st.update(quality=q)
        (wd / f"{vid}.info.json").write_text(json.dumps({"id": vid, "title": inf["title"], "duration": inf["duration"],
                                                          "uploader": inf["author"], "recorded": st}, ensure_ascii=False,
                                                         default=str), encoding="utf-8")
        log(f"  recorded {st['w']}x{st['h']}  {st['frames']} frames ({st['missing']} missing)  ads skipped "
            f"{st['ad_s']:.0f}s  sound pauses {st['seams']}")
        return st

    return out, _with_routes(cfg, go, log)


# recording cost, seconds of work per second of clip (picture + sound + seeks), learned while running. Starting
# guesses from a 4-core desktop; the first clip replaces them with the real number for this machine.
_COST = {"hd2160": 6.0, "hd1440": 3.5, "hd1080": 2.6, "hd720": 2.2, "large": 2.0, "medium": 2.0, "small": 2.0,
         "tiny": 2.0, "auto": 2.0}
_BUDGET = {"left_s": None, "left": 0}           # set by daily.post: every clip of today's run shares one RECORDING budget


def set_budget(seconds: float, clips: int) -> None:
    _BUDGET.update(left_s=float(seconds), left=int(clips))


def _pick_quality(ladder: list[str], k: int, dur: float, log=print) -> int:
    """Best quality whose expected cost still fits this clip's share of the remaining budget (never below k...
    except when the budget forces it)."""
    if _BUDGET["left_s"] is None:
        return k
    share = _BUDGET["left_s"] / max(1, _BUDGET["left"])           # recording time only (renders don't count)
    for i, q in enumerate(ladder):
        if _COST.get(q, 3.0) * dur <= share or i == len(ladder) - 1:
            if i != k:
                log(f"    time budget: {share / 60:.1f} min per clip left -> {q} "
                    f"(costs ~{_COST.get(q, 3.0) * dur / 60:.1f} min here)")
            return i
    return k


def record_clips(cfg, vid: str, spans: list[tuple[float, float, Path]], log=print) -> list[Path]:
    """Each span: picture at the best quality offered (4K first) with no missing frame + an unbroken 1x sound take."""
    r = _rc(cfg)
    lo = QUALITIES.index(r["clip_min_quality"])
    todo = [s for s in spans if not s[2].exists()]
    if not todo:
        return [s[2] for s in spans]

    def go(br):
        br.open(vid)
        av = br.info()["qualities"]
        ladder = [x for x in QUALITIES[QUALITIES.index(r["clip_max_quality"]): lo + 1] if x in av] or ["auto"]
        log(f"  recording {len(todo)} clips in Chrome (offered: {', '.join(x for x in av if x != 'auto')}) - no download")
        k, rate = 0, None
        lim_gap, lim_frac = int(r["max_gap_frames"]), float(r["max_missing_frac"])
        for a, b, out in todo:
            b += float(r["tail_s"])             # a little past the end: the render needs the span's last frame too
            tv, ta = out.with_suffix(".v.ivf"), out.with_suffix(".a.webm")
            best = out.with_suffix(".best.ivf")
            t0 = time.time()
            k2 = _pick_quality(ladder, k, b - a, log)
            if k2 != k:
                k, rate = k2, None
            best_st, tries_at_min = None, 0
            while True:                                   # picture
                q = ladder[k]
                h = HEIGHT.get(q, 1080)
                rt = rate or (0.25 if h > 1440 else 0.5 if h > 720 else 1.0)   # this PC: 1080p60 at 1x drops 23 %
                try:
                    st = br.record_video(vid, a, b, tv, q, rt, log)
                except Stuck as e:
                    if k == len(ladder) - 1:
                        if best_st:                       # keep the best take we already have
                            st = best_st[1]
                            break
                        raise
                    k, rate, tries_at_min = k + 1, None, 0
                    log(f"    {e} -> trying {ladder[k]}")
                    continue
                log(f"    {out.name}: picture {st['w']}x{st['h']} ({st['quality']}, {st['codec']}) at {rt}x speed, "
                    f"{st['frames']} frames, {st['missing']} missing (longest gap {st['max_gap']}), encoder waits {st['throttles']}, "
                    f"skips repaired {st.get('repairs', 0)}")
                score = (st["max_gap"] > lim_gap, st["missing_frac"])
                if best_st is None or score < best_st[0]:  # keep the best take so far (a retry can be worse)
                    os.replace(tv, best)
                    best_st = (score, st)
                if st["missing_frac"] <= lim_frac and st["max_gap"] <= lim_gap:
                    rate = rt
                    break
                if rt / 2 >= r["min_rate"]:
                    rate = rt / 2
                    log(f"    frames were missing -> again at {rate}x speed")
                elif tries_at_min < 1:
                    tries_at_min += 1                     # slowest speed: one more take before giving up quality
                    rate = rt
                    log(f"    frames were missing -> one more take at {rt}x")
                elif k < len(ladder) - 1 and best_st[0][0]:
                    k, tries_at_min = k + 1, 0            # only real hitches (long gaps) cost quality; speed stays slow
                    log(f"    hitches at {rt}x -> {ladder[k]}")
                else:
                    break
            if best_st[1] is not st or not best.exists():
                st = best_st[1]
            if best.exists():
                os.replace(best, tv)
            if not (st["missing_frac"] <= lim_frac and st["max_gap"] <= lim_gap):
                log(f"    keeping the best take: {st['h']}p, {st['missing']} single missing frames, longest gap {st['max_gap']}")
            for _ in range(int(r["audio_retries"])):     # sound: one unbroken 1x take
                try:
                    sa = br.record_audio(vid, a, b, ta, log)
                except Stuck as e:
                    log(f"    sound: {e} -> again")
                    continue
                if sa["seams"] == 0:
                    break
                log(f"    sound had {sa['seams']} pause(s) (ad / buffering) -> recording the sound again")
            else:
                log("    WARNING: sound still had pauses after retries; keeping the last take")
            br.mux(tv, ta, out, st.get("lead_s", 0.0))
            tv.unlink(missing_ok=True)
            ta.unlink(missing_ok=True)
            el = time.time() - t0
            _COST[st["quality"]] = 0.5 * _COST.get(st["quality"], el / (b - a)) + 0.5 * el / (b - a)
            _BUDGET["left"] = max(0, _BUDGET["left"] - 1)
            if _BUDGET["left_s"] is not None:
                _BUDGET["left_s"] -= el
            log(f"    {out.name} done in {el:.0f} s ({st['h']}p)")
        return [s[2] for s in spans]

    return _with_routes(cfg, go, log)
