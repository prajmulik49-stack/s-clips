"""Channel-correct metadata research and Short-to-vlog confirmation."""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import median

from . import ingest, tools
from .text import hinglish, lex

GENERIC = {"vlog", "short", "shorts", "official", "fan", "new", "today", "guys", "video",
           "the", "and", "with", "this", "that", "mera", "meri", "mere", "ka", "ki", "ke", "hai"}


def _date(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, dt.timezone.utc)
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            x = dt.datetime.strptime(str(value), fmt)
            return x.replace(tzinfo=x.tzinfo or dt.timezone.utc)
        except ValueError:
            continue
    return None


def _iso(value):
    x = _date(value)
    return x.astimezone(dt.timezone.utc).isoformat() if x else None


def _record(raw, channel, role, source_url):
    timestamp = raw.get("timestamp") or raw.get("release_timestamp") or raw.get("upload_date")
    return {
        "id": raw.get("id") or raw.get("url"),
        "title": (raw.get("title") or "").strip(),
        "channel": channel,
        "role": role,
        "url": raw.get("webpage_url") or raw.get("original_url") or raw.get("url") or source_url,
        "upload_time": _iso(timestamp),
        "duration_s": float(raw["duration"]) if raw.get("duration") is not None else None,
        "views": int(raw["view_count"]) if raw.get("view_count") is not None else None,
        "description": raw.get("description") or "",
    }


def _yt_json(cfg, url, flat=False, playlist_end=None):
    args = ingest.ytdlp_base(cfg) + ["--dump-json", "--skip-download"]
    if flat:
        args += ["--flat-playlist"]
    if playlist_end:
        args += ["--playlist-end", str(playlist_end)]
    p = subprocess.run(args + [url], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=tools.child_env(), check=False)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "yt-dlp failed").strip()[-1200:])
    result = []
    for line in p.stdout.splitlines():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(x, dict) and (x.get("id") or x.get("url")):
            result.append(x)
    return result


def channel_sources(cfg):
    ch = cfg["channel"]
    cid = ch["official_id"]
    sources = [{"channel": "official_vlogs", "role": "vlog",
                "url": ch.get("official_vlogs_url") or f"https://www.youtube.com/playlist?list=UULF{cid[2:]}"},
               {"channel": "official_shorts", "role": "official_short",
                "url": ch.get("official_shorts_url", "https://www.youtube.com/@souravjocshorts/shorts")}]
    urls = ch.get("fan_urls") or [f"https://www.youtube.com/channel/{x}/shorts" for x in ch.get("fan_ids", [])]
    for i, url in enumerate(urls):
        sources.append({"channel": f"fan_{i + 1}", "role": "fan_short", "url": url})
    for i, playlist in enumerate(ch.get("fan_shorts_playlists", [])):
        sources.append({"channel": f"fan_{i + 1}", "role": "fan_short",
                        "url": f"https://www.youtube.com/playlist?list={playlist}"})
    return sources


def collect_catalog(cfg, limit=None, log=print):
    """Metadata-only pass. Rich metadata is fetched after cheap flat enumeration."""
    limit = limit or int(cfg["research"].get("latest_vlogs", 60))
    groups = {"vlogs": [], "official_shorts": [], "fan_shorts": []}
    for source in channel_sources(cfg):
        log(f"  catalog {source['channel']} ({source['role']})")
        flat = _yt_json(cfg, source["url"], flat=True, playlist_end=limit if source["role"] == "vlog" else None)
        seen = set()
        for item in flat:
            vid = item.get("id") or item.get("url")
            if vid in seen:
                continue
            seen.add(vid)
            page = item.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
            try:
                rich = _yt_json(cfg, page)[0]
            except Exception as exc:
                log(f"    metadata unavailable for {vid}: {exc}")
                rich = item
            rec = _record(rich, source["channel"], source["role"], page)
            key = "vlogs" if source["role"] == "vlog" else "official_shorts" if source["role"] == "official_short" else "fan_shorts"
            if rec["id"]:
                groups[key].append(rec)
    for key in groups:
        groups[key] = list({x["id"]: x for x in groups[key]}.values())
        groups[key].sort(key=lambda x: x.get("upload_time") or "", reverse=True)
    groups["vlogs"] = groups["vlogs"][:limit]
    return {"schema": 1, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), **groups}


def _tokens(title):
    words = re.findall(r"[\w]+", hinglish.romanize(title).lower(), flags=re.UNICODE)
    return {lex.key_word(w) for w in words if len(w) >= 3 and w not in GENERIC and lex.key_word(w)}


def cheap_pairs(catalog, cfg):
    """Pair on same/next-day upload and title-keyword overlap before media work."""
    max_days = int(cfg["research"].get("pair_max_day_delta", 1))
    minimum = int(cfg["research"].get("pair_min_keyword_overlap", 1))
    out = []
    for short in catalog.get("official_shorts", []) + catalog.get("fan_shorts", []):
        st = _date(short.get("upload_time"))
        choices = []
        for vlog in catalog.get("vlogs", []):
            vt = _date(vlog.get("upload_time"))
            if not st or not vt:
                continue
            delta = abs((st.date() - vt.date()).days)
            overlap = _tokens(short.get("title", "")) & _tokens(vlog.get("title", ""))
            if delta <= max_days and len(overlap) >= minimum:
                score = 0.55 * min(1.0, len(overlap) / max(3, len(_tokens(short.get("title", ""))))) + 0.45 * (1 - delta / max(1, max_days + 1))
                choices.append((score, -delta, len(overlap), vlog, overlap))
        if choices:
            score, neg_delta, _, vlog, overlap = max(choices, key=lambda x: x[:3])
            out.append({"short_id": short["id"], "vlog_id": vlog["id"], "source": short["role"],
                        "cheap_score": round(score, 4), "day_delta": -neg_delta,
                        "keyword_overlap": sorted(overlap), "confirmed": False})
    return out


def _audio_env(x, hop):
    import numpy as np
    n = len(x) // hop
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10).astype(np.float32)


def piecewise_audio_align(short_path, vlog_path, approx_offset=0.0, search_s=180.0,
                          sample_rate=8000, block_s=2.0):
    """Return vlog source spans for each Short block; jumps expose internal edits."""
    import numpy as np
    from scipy.signal import fftconvolve
    from . import media
    short = media.read_audio(Path(short_path), sample_rate).astype(np.float32)
    vlog = media.read_audio(Path(vlog_path), sample_rate).astype(np.float32)
    if len(short) < sample_rate or len(vlog) < sample_rate:
        return {"confirmed": False, "confidence": 0.0, "segments": [], "internal_cuts": 0}
    step_s = 0.02
    hop = max(1, int(sample_rate * step_s))
    se, ve = _audio_env(short, hop), _audio_env(vlog, hop)
    block = max(1, int(block_s / step_s))
    matches = []
    for a in range(0, max(1, len(se) - block + 1), max(1, block // 2)):
        q = se[a:a + block]
        if len(q) < max(8, block // 2):
            continue
        lo = max(0, int((approx_offset - search_s) / step_s))
        hi = min(max(0, len(ve) - len(q)), int((approx_offset + search_s) / step_s) if approx_offset else len(ve) - len(q))
        if hi < lo:
            lo, hi = 0, max(0, len(ve) - len(q))
        r = ve[lo:hi + len(q)]
        q = (q - q.mean()) / (q.std() + 1e-6)
        r = (r - r.mean()) / (r.std() + 1e-6)
        corr = fftconvolve(r, q[::-1], mode="valid") / max(1, len(q))
        k = int(np.argmax(corr)) if len(corr) else 0
        matches.append({"short_s": a * step_s, "short_e": min(len(se), a + block) * step_s,
                        "source_s": (lo + k) * step_s, "source_e": (lo + k + len(q)) * step_s,
                        "confidence": float(np.clip(corr[k] if len(corr) else 0, 0, 1))})
    good = [x for x in matches if x["confidence"] >= 0.35]
    segments = []
    for item in good:
        if not segments:
            segments.append(dict(item))
            continue
        prev = segments[-1]
        if abs((item["source_s"] - prev["source_e"]) - (item["short_s"] - prev["short_e"])) <= 0.35:
            prev["short_e"], prev["source_e"] = item["short_e"], item["source_e"]
            prev["confidence"] = min(prev["confidence"], item["confidence"])
        else:
            segments.append(dict(item))
    confidence = float(np.mean([x["confidence"] for x in good])) if good else 0.0
    return {"confirmed": bool(segments and confidence >= 0.45), "confidence": round(confidence, 4),
            "segments": segments, "internal_cuts": max(0, len(segments) - 1)}


def apply_audio_confirmation(catalog, pairs, media_paths, cfg, log=print):
    for pair in pairs:
        short, vlog = media_paths.get(pair["short_id"]), media_paths.get(pair["vlog_id"])
        if not short or not vlog:
            pair["confirmation_error"] = "media_not_available"
            continue
        try:
            result = piecewise_audio_align(short, vlog, search_s=180,
                sample_rate=int(cfg["research"].get("audio_sample_rate", 8000)),
                block_s=float(cfg["research"].get("audio_block_s", 2.0)))
            pair.update({"confirmed": result["confirmed"], "audio_confidence": result["confidence"],
                         "source_segments": result["segments"], "internal_cuts": result["internal_cuts"]})
        except Exception as exc:
            pair["confirmation_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  audio confirmation failed for {pair['short_id']}: {pair['confirmation_error']}")
    return pairs


def categorize(title, cfg):
    low = hinglish.romanize(title).lower()
    scores = {cat: sum(1 for phrase in phrases if lex.phrase_in(str(phrase), low, 75))
              for cat, phrases in cfg["research"].get("category_patterns", {}).items()}
    best = max(scores, key=lambda x: (scores[x], x)) if scores else "montage"
    return best if scores.get(best, 0) else "montage"


def mine_title_grammar(shorts):
    names = {str(x).lower() for x in (lex.lexicon().get("names") or [])}
    official, fan = collections.Counter(), collections.Counter()
    for item in shorts:
        tokens = re.findall(r"[\w]+", hinglish.romanize(item.get("title", "")).lower())
        if not tokens:
            continue
        if item["role"] == "official_short":
            official[("who" if tokens[0] in names else "cue", "what", "gaya/ho_gaya", "emoji")] += 1
        else:
            fan[("who", "ne", "what", "object", "emoji_pair")] += 1
    return {"official": [{"template": list(k), "count": n} for k, n in official.most_common()],
            "fan": [{"template": list(k), "count": n} for k, n in fan.most_common()]}


def build_report(catalog, pairs, cfg, as_of=None):
    as_of = as_of or dt.datetime.now(dt.timezone.utc)
    shorts = catalog.get("official_shorts", []) + catalog.get("fan_shorts", [])
    for item in shorts:
        item["category"] = categorize(item.get("title", ""), cfg)
        t = _date(item.get("upload_time"))
        item["age_days"] = max(1, (as_of - t).total_seconds() / 86400) if t else 1
    by_id = {x["id"]: x for x in shorts}
    channel_values = collections.defaultdict(list)
    for x in shorts:
        if x.get("views") is not None:
            channel_values[x["role"]].append(x["views"] / x["age_days"])
    category_values = collections.defaultdict(list)
    for x in shorts:
        if x.get("views") is None:
            continue
        base = median(channel_values[x["role"]]) if channel_values[x["role"]] else 1
        x["normalized_views"] = (x["views"] / x["age_days"]) / max(1, base)
        category_values[(x["role"], x["category"])].append(x["normalized_views"])
    per_vlog = {}
    for vlog in catalog.get("vlogs", []):
        ps = [p for p in pairs if p["vlog_id"] == vlog["id"]]
        official = [p for p in ps if p["source"] == "official_short"]
        fan = [p for p in ps if p["source"] == "fan_short"]
        consensus = []
        for a in official:
            for b in fan:
                for sa in a.get("source_segments", []):
                    for sb in b.get("source_segments", []):
                        inter = max(0, min(sa["source_e"], sb["source_e"]) - max(sa["source_s"], sb["source_s"]))
                        den = min(sa["source_e"] - sa["source_s"], sb["source_e"] - sb["source_s"])
                        if den and inter / den >= cfg["research"].get("consensus_overlap_frac", 0.35):
                            consensus.append({"official_short_id": a["short_id"], "fan_short_id": b["short_id"],
                                              "source_s": max(sa["source_s"], sb["source_s"]), "source_e": min(sa["source_e"], sb["source_e"])})
        durations = [by_id[p["short_id"]]["duration_s"] for p in ps if by_id.get(p["short_id"], {}).get("duration_s")]
        delays = []
        vt = _date(vlog.get("upload_time"))
        for p in ps:
            st = _date(by_id[p["short_id"]].get("upload_time"))
            if st and vt:
                delays.append((st - vt).total_seconds() / 3600)
        per_vlog[vlog["id"]] = {"title": vlog["title"], "official_count": len(official), "fan_count": len(fan),
            "consensus_count": len(consensus), "consensus": consensus,
            "short_duration_s": {"n": len(durations), "min": min(durations) if durations else None,
                                 "median": median(durations) if durations else None, "max": max(durations) if durations else None},
            "short_to_vlog_delay_h": {"n": len(delays), "median": median(delays) if delays else None,
                                      "min": min(delays) if delays else None, "max": max(delays) if delays else None}}
    cats = {}
    for (role, category), values in sorted(category_values.items()):
        cats.setdefault(role, {})[category] = {"n": len(values), "median_normalized_views": median(values)}
    return {"schema": 1, "as_of": as_of.isoformat(), "vlogs": per_vlog, "categories": cats,
            "title_grammar": mine_title_grammar(shorts), "pairs": pairs}


def write_outputs(cfg, catalog, report, log=print):
    directory = cfg.path("research")
    directory.mkdir(parents=True, exist_ok=True)
    catalog_path = directory / cfg["research"].get("catalog_file", "catalog.json")
    report_path = directory / cfg["research"].get("report_file", "report.json")
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    lines = ["# JoshiClip research", "",
             f"Vlogs: {len(catalog.get('vlogs', []))} · official Shorts: {len(catalog.get('official_shorts', []))} · fan Shorts: {len(catalog.get('fan_shorts', []))}",
             "", "## Per-vlog counts", "", "| vlog | official | fan | consensus | median duration | median delay |", "|---|---:|---:|---:|---:|---:|"]
    for vid, item in report["vlogs"].items():
        lines.append(f"| {item['title']} ({vid}) | {item['official_count']} | {item['fan_count']} | {item['consensus_count']} | {item['short_duration_s']['median'] or '-'} | {item['short_to_vlog_delay_h']['median'] or '-'} h |")
    lines += ["", "## Categories", "", "json", json.dumps(report["categories"], ensure_ascii=False, indent=2), "",
              "## Title grammar", "", "json", json.dumps(report["title_grammar"], ensure_ascii=False, indent=2), ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"  research: {report_path}")
    return catalog_path, report_path


def quick(cfg, limit=400, log=print):
    """Flat listings only (title + view count, ~1 min): median-views lift per category -> research/report.json
    `category_lift`, which moment_first uses as its "what kind of moment performs" prior. Shorts tabs list newest
    first, so each channel is compared against its own median over the same window."""
    shorts, failed = [], []
    for source in channel_sources(cfg):
        if source["role"] == "vlog":
            continue
        log(f"  quick catalog {source['channel']}")
        rows = None
        for attempt in range(3):                       # network blips ("getaddrinfo failed") -> retry, then skip
            try:
                rows = _yt_json(cfg, source["url"], flat=True, playlist_end=limit)
                break
            except RuntimeError as exc:
                log(f"    attempt {attempt + 1} failed: {str(exc).splitlines()[-1][:160]}")
                time.sleep(10 * (attempt + 1))
        if rows is None:
            log(f"    skipped {source['channel']} (no connection); its old numbers stay")
            failed.append(source["channel"])
            continue
        for raw in rows:
            if raw.get("view_count") is not None and raw.get("title"):
                shorts.append({"id": raw.get("id"), "title": raw["title"], "views": int(raw["view_count"]),
                               "role": source["role"], "channel": source["channel"],
                               "category": categorize(raw["title"], cfg)})
    shorts = list({x["id"]: x for x in shorts}.values())
    if failed:                                      # half the data would skew the priors -> keep the old report
        log(f"  research (quick): could not reach {', '.join(failed)}; report.json unchanged. Check the internet and rerun.")
        return 1
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    base = collections.defaultdict(list)
    for x in shorts:
        base[x["channel"]].append(x["views"])
    for x in shorts:
        per[x["category"]][x["channel"]].append(x["views"] / max(1, median(base[x["channel"]])))
    lift, table = {}, {}
    for cat, chans in per.items():
        vals = [median(v) for v in chans.values() if len(v) >= 5]       # each channel weighs the same
        table[cat] = {ch: {"n": len(v), "lift": round(median(v), 3)} for ch, v in chans.items()}
        if vals:
            lift[cat] = round(sum(vals) / len(vals), 3)
    directory = cfg.path("research")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / cfg["research"].get("report_file", "report.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {"schema": 1}
    report.update({"category_lift": lift, "category_lift_detail": table,
                   "category_lift_as_of": dt.datetime.now(dt.timezone.utc).isoformat(), "category_lift_n": len(shorts)})
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    for cat, v in sorted(lift.items(), key=lambda kv: -kv[1]):
        log(f"  {cat:20s} {v:.2f}x median views")
    log(f"  research (quick, {len(shorts)} Shorts): {path}")
    return 0


def run(cfg, limit=None, log=print):
    catalog = collect_catalog(cfg, limit, log)
    pairs = cheap_pairs(catalog, cfg)
    write_outputs(cfg, catalog, build_report(catalog, pairs, cfg), log)
    return 0
