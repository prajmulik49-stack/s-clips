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

    p = sub.add_parser("explain"); _common(p); p.add_argument("report")

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
        from .research import run
        return run(cfg, a.limit)
    if a.cmd == "run":
        mode = "final" if a.final else "draft"
        if a.fast:
            from .fast import run_fast
            return run_fast(cfg, a.target, a.latest, a.id, mode=mode if (a.final or a.draft) else "final", max_shorts=a.max_shorts)
        from .pipeline import run
        return run(cfg, a.target, a.latest, a.id, mode=mode, max_shorts=a.max_shorts,
                   export_candidates=a.export_candidates, preview_debug=a.preview_debug, render=not a.no_render,
                   plain=a.plain)
    if a.cmd == "explain":
        from .report import explain
        return explain(a.report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
