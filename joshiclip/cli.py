"""joshiclip command line."""
from __future__ import annotations

import argparse
import sys

from . import config as config_mod


def _common(p):
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--id", default=None, help="explicit video id (for local files)")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="joshiclip")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor"); _common(p)

    p = sub.add_parser("ingest", help="download + normalize only"); _common(p)
    p.add_argument("target", nargs="?"); p.add_argument("--latest", action="store_true")

    p = sub.add_parser("analyze", help="ingest + full-video analysis (cached)"); _common(p)
    p.add_argument("target", nargs="?"); p.add_argument("--latest", action="store_true")

    p = sub.add_parser("bench-asr", help="benchmark Whisper models on an excerpt"); _common(p)
    p.add_argument("target")

    p = sub.add_parser("research", help="catalog the latest vlogs and both Shorts sources"); _common(p)
    p.add_argument("--limit", type=int, default=None, help="number of latest vlogs (default: config research.latest_vlogs)")
    p.add_argument("--quick", action="store_true", help="titles + views only (~1 min): refresh the category priors the scorer uses")

    p = sub.add_parser("run", help="full pipeline"); _common(p)
    p.add_argument("target", nargs="?"); p.add_argument("--latest", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--draft", action="store_true"); g.add_argument("--final", action="store_true")
    p.add_argument("--max-shorts", type=int, default=None)
    p.add_argument("--export-candidates", action="store_true")
    p.add_argument("--preview-debug", action="store_true")
    p.add_argument("--no-render", action="store_true", help="plan + report only")
    p.add_argument("--plain", action="store_true", help="plain 16:9 cuts instead of the vertical layout")
    p.add_argument("--fast", action="store_true", help="360p analysis + YouTube captions + 4K only for winners (~10-15 min)")

    p = sub.add_parser("reaudio", help="redo only the audio (voice settings) of rendered fast-mode Shorts"); _common(p)
    p.add_argument("video_id")

    p = sub.add_parser("explain"); _common(p); p.add_argument("report")

    p = sub.add_parser("daily", help="plan / render / upload today's Shorts for every channel"); _common(p)
    p.add_argument("action", choices=["plan", "post", "run"])
    p.add_argument("--channel", default=None, help="post: channel name from config daily.channels")
    p.add_argument("--plan", default=None, help="post: plan json (default state/plans/<today>.json)")
    p.add_argument("--dry-run", action="store_true", help="plan only, print it, save nothing")
    p.add_argument("--no-upload", action="store_true", help="render but do not upload")
    p.add_argument("--replan", action="store_true", help="make a new plan even if today's exists")

    p = sub.add_parser("auth", help="one-time YouTube login for a channel -> prints its GitHub secrets"); _common(p)
    p.add_argument("client_secret", nargs="?", default=None,
                   help="OAuth client JSON from Google Cloud (Desktop app); default: YOUTUBE_OAUTH_CLIENT_JSON in .env")
    p.add_argument("--prefix", default="YT_A", help="secret prefix of the channel (config daily.channels env)")

    p = sub.add_parser("check-auth", help="test every channel's YouTube login (fails if a token will expire)"); _common(p)
    p.add_argument("--channel", default=None, help="only this channel (config daily.channels name)")

    a = ap.parse_args(argv)
    cfg = config_mod.load(a.config)

    if a.cmd == "doctor":
        from .doctor import doctor
        return doctor(cfg)
    if a.cmd == "ingest":
        from . import ingest
        from .pipeline import workdir_for
        vid, src = ingest.resolve_source(cfg, a.target, a.latest)
        vid = a.id or vid
        meta = ingest.normalize(cfg, src, workdir_for(cfg, vid))
        print(f"{vid}: {meta['fps']} fps, {meta['width']}x{meta['height']}, {meta['duration']:.1f}s  checks ok={meta['checks']['ok']}")
        return 0
    if a.cmd == "analyze":
        from .pipeline import analyze
        analyze(cfg, a.target, a.latest, a.id)
        return 0
    if a.cmd == "bench-asr":
        from .analysis.transcript import bench
        from .pipeline import workdir_for
        from . import ingest
        vid, src = ingest.resolve_source(cfg, a.target, False)
        bench(cfg, workdir_for(cfg, a.id or vid))
        return 0
    if a.cmd == "research":
        from .text import lex
        lex.set_lexicon(config_mod.load_lexicon(cfg))
        from .research import quick, run
        return quick(cfg, a.limit or 400) if a.quick else run(cfg, a.limit)
    if a.cmd == "run":
        mode = "final" if a.final else "draft"
        if a.fast:
            from .fast import run_fast
            return run_fast(cfg, a.target, a.latest, a.id, mode=mode if (a.final or a.draft) else "final", max_shorts=a.max_shorts)
        from .pipeline import run
        return run(cfg, a.target, a.latest, a.id, mode=mode, max_shorts=a.max_shorts,
                   export_candidates=a.export_candidates, preview_debug=a.preview_debug, render=not a.no_render,
                   plain=a.plain)
    if a.cmd == "reaudio":
        from .fast import reaudio
        return reaudio(cfg, a.video_id)
    if a.cmd == "daily":
        from . import daily
        if a.action == "plan":
            daily.plan_day(cfg, a.dry_run, print, a.replan)
            return 0
        if a.action == "post":
            import json
            from pathlib import Path
            if not a.channel:
                ap.error("daily post needs --channel")
            plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
            return daily.post(cfg, a.channel, plan, not a.no_upload)
        return daily.run(cfg, a.dry_run, not a.no_upload, print, a.replan)
    if a.cmd == "auth":
        from .youtube import auth
        return auth(a.client_secret, a.prefix)
    if a.cmd == "check-auth":
        from .youtube import check_all
        return check_all(cfg, a.channel)
    if a.cmd == "explain":
        from .report import explain
        return explain(a.report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
