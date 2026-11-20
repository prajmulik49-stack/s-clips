"""upload.md / upload.json: what to paste into YouTube for every Short, best Short first.

The burned-in header is only half the packaging: the upload title, the first description line and the hashtags decide
search / suggested reach. Everything here comes from the plan (title, category, names) - no generated text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CATEGORY_TAGS = {"funny_accident": ["#funny", "#comedy"], "danger_loss_fear": ["#shocking", "#omg"],
                 "festival_emotional": ["#emotional", "#family"], "drama_relationship": ["#family", "#drama"],
                 "family_home": ["#family", "#saasbahu"], "everyday": ["#family"],
                 "dance": ["#dance"], "money_numbers": ["#money"], "travel_scenery": ["#travel"]}
# Name -> the hashtag fans actually search (the top fan channel tags every Avantika clip #avantikabhatt).
NAME_TAGS = {"avantika": "#avantikabhatt", "sourav": "#souravjoshi"}


def _clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def build(rep: dict, cfg, vlog_title: str | None) -> list[dict]:
    u = cfg.get("upload", {})
    from .text import lex
    names = {str(n).lower(): str(n) for n in (lex.lexicon().get("names") or [])}
    vid = rep["video_id"]
    items = []
    for s in sorted(rep["shorts"], key=lambda s: (-s["final"], s["start"])):
        head = s["title"].strip()
        words = re.findall(r"[A-Za-z]+", head)
        who = sorted({names[w.lower()] for w in words if w.lower() in names})
        tags = list(u.get("channel_hashtags", ["#shorts"]))
        for w in who:
            tag = NAME_TAGS.get(w.lower(), f"#{w.lower()}")
            if tag not in tags:
                tags.append(tag)
        tags += [t for t in CATEGORY_TAGS.get(s["tag"], []) if t not in tags]
        tags = tags[: int(u.get("max_hashtags", 6))]
        yt_title = f"{head} {u.get('title_suffix', '')}".strip()
        if len(yt_title) > 100:
            yt_title = head[:100]
        link = f"https://youtu.be/{vid}?t={int(s['start'])}"
        desc = [f"{head}", "", f"Full vlog{(': ' + vlog_title) if vlog_title else ''} -> {link}"]
        if u.get("credit_line"):
            desc.append(u["credit_line"])
        desc += ["", " ".join(tags)]
        items.append({"rank": len(items) + 1, "file": s["file"], "title": yt_title, "description": "\n".join(desc),
                      "hashtags": tags, "source": f"{_clock(s['start'])}-{_clock(s['end'])}", "score": s["final"],
                      "title_source": s.get("title_source"), "viewer_score": s.get("viewer_score")})
    return items


def write(rep: dict, cfg, out_dir: Path, vlog_title: str | None) -> Path:
    items = build(rep, cfg, vlog_title)
    (out_dir / "upload.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Upload pack", "", "Post in this order (best first). Post soon after the vlog goes live; that's when fans search for it.", ""]
    for it in items:
        L += [f"## {it['rank']}. {it['file']}", "", f"- source: {it['source']} · score {it['score']:.3f}"
              + (f" · viewers {it['viewer_score']:.2f}" if it.get("viewer_score") else "")
              + (f" · title from a viewer comment" if it.get("title_source") == "viewer_comment" else ""),
              "", "**Title**", "", "```", it["title"], "```", "", "**Description**", "", "```", it["description"], "```", ""]
    p = out_dir / "upload.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
