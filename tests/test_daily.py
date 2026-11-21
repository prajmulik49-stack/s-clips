import datetime as dt
import json

from joshiclip import daily, ledger, music


def _state(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "STATE", tmp_path / "state")
    return tmp_path / "state"


def test_any_overlap_in_any_channel_blocks_the_scene(tmp_path, monkeypatch):
    _state(tmp_path, monkeypatch)
    ledger.record_post("channel_a", {"channel": "channel_a", "vid": "v1", "start": 100.0, "end": 140.0, "title": "T"})
    ledger.save_plan({"date": "2026-11-20", "channels": {"channel_b": [{"vid": "v2", "start": 10.0, "end": 50.0}]}})
    u = ledger.used_spans("2026-11-21", reserve_days=2)
    assert ledger.overlaps(u, "v1", 139.9, 170)            # 0.1 s shared with channel_a's post
    assert not ledger.overlaps(u, "v1", 140.0, 170)        # touching is fine
    assert ledger.overlaps(u, "v2", 0, 11)                 # reserved by yesterday's plan for channel_b
    assert not ledger.overlaps(ledger.used_spans("2026-11-25", 2), "v2", 0, 11)   # reservation expired
    assert ledger.used_titles() == {"T"}


def test_assign_round_robin_and_caps():
    ms = [{"final": 1 - i / 10, "i": i} for i in range(10)]
    got = daily.assign(ms, {"a": 2, "b": 3})
    assert [m["i"] for m in got["a"]] == [0, 2] and [m["i"] for m in got["b"]] == [1, 3, 4]
    got = daily.assign(ms, {"a": 5, "b": 5}, cap=4)
    assert sum(len(v) for v in got.values()) == 4


def test_slots_ist_and_rollover():
    d = {"publish_times": ["08:00", "21:30"], "channel_offset_min": [0, 40], "schedule": True}
    ist = dt.timezone(dt.timedelta(minutes=330))
    now = dt.datetime(2026, 11, 21, 2, 0, tzinfo=ist)
    assert daily._slot(d, 0, 0, now) == "2026-11-21T02:30:00Z"     # 08:00 IST
    assert daily._slot(d, 1, 1, now) == "2026-11-21T16:40:00Z"     # 21:30 + 40 min IST
    late = dt.datetime(2026, 11, 21, 22, 0, tzinfo=ist)
    assert daily._slot(d, 0, 0, late) == "2026-11-22T02:30:00Z"    # past -> tomorrow


def test_music_pick_is_seeded(tmp_path):
    for n in ("a.mp3", "b.wav", "c.m4a", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    p1 = music.pick(tmp_path, "vid:12.000")
    assert p1 == music.pick(tmp_path, "vid:12.000") and p1.suffix != ".txt"
    assert len({music.pick(tmp_path, f"vid:{i}").name for i in range(40)}) == 3
    assert music.pick(tmp_path / "missing", "x") is None


def test_planned_scene_is_found_by_span_when_ids_shift():
    from joshiclip import fast
    cands = [{"id": i, "start": 10.0 * i, "end": 10.0 * i + 40} for i in range(5)]
    assert fast._match_pick(cands, [1, 3], [(10, 50), (30, 70)], lambda *a: None) == [1, 3]   # unchanged
    assert fast._match_pick(cands, [0, 4], [(20, 60), (999, 1039)], lambda *a: None) == [2]   # moved / gone
    assert fast._match_pick(cands, [3], None) == [3]                                          # old callers


def test_prune_cache_drops_oldest_but_keeps_planned(tmp_path):
    import os
    from joshiclip.config import Config
    cfg = Config({"paths": {"work": str(tmp_path)}, "daily": {}})
    for k, name in enumerate(["old", "mid", "new", "full"]):
        d = tmp_path / (name if name == "full" else f"{name}-fast")
        d.mkdir()
        (d / "norm.json").write_bytes(b"x" * 1000)
        os.utime(d / "norm.json", (1000 + k, 1000 + k))
    gone = daily.prune_cache(cfg, keep={"old"}, max_gb=2000 / 1024 ** 3, log=lambda *a: None)
    assert gone == ["mid-fast"]                                  # oldest unplanned goes; "old" is planned today
    assert (tmp_path / "old-fast").exists() and (tmp_path / "new-fast").exists() and (tmp_path / "full").exists()


def test_uploaded_short_and_sources_are_deleted(tmp_path):
    for p in ["01_A.mp4", "01_A/x.mp4", "short_01/voice.wav", "02_B.mp4", "short_02/voice.wav",
              "clip_01.mkv", "clip_02.v.ivf", "source.info.json", "report.md", "upload.md"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True); (tmp_path / p).write_text("x")
    daily._delete_uploaded(tmp_path, "01_A.mp4")
    daily._delete_sources(tmp_path, log=lambda *a: None)
    left = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*") if p.is_file())
    assert left == ["02_B.mp4", "report.md", "short_02/voice.wav", "upload.md"]   # 02 wasn't uploaded


def test_cta_phrases_match_hindi_captions_without_false_hits():
    import yaml
    from joshiclip.config import ROOT
    from joshiclip.text import lex
    lex.set_lexicon(yaml.safe_load((ROOT / "config/lexicon.yaml").read_text(encoding="utf-8")))
    assert lex.hits("cta", "पसंद आया तो लाइक कर देना") and lex.hits("cta", "चैनल पे सब्सक्राइब कर देना")
    assert lex.hits("cta", "कमेंट करके जरूर बताना")
    assert not lex.hits("cta", "आज हम एयरपोर्ट जा रहे हैं")
    assert not lex.hits("intro_outro", "हां") and not lex.hits("intro_outro", "ये")   # one-word sentences


def test_old_vlogs_one_cached_then_one_random(tmp_path):
    import random
    from joshiclip.config import Config
    cfg = Config({"paths": {"work": str(tmp_path)}, "daily": {}})
    for v in ("c1", "c2"):
        (tmp_path / f"{v}-fast").mkdir(); (tmp_path / f"{v}-fast" / "norm.json").write_text("{}")
    old = [{"vid": v} for v in ("n1", "c1", "n2", "x", "c2", "n3")]
    got = [v["vid"] for v in daily.old_vlog_order(cfg, old, {"x"}, random.Random(1))]
    assert got[0] in ("c1", "c2") and got[1] in ("n1", "n2", "n3") and "x" not in got and len(got) == 5
    assert got == [v["vid"] for v in daily.old_vlog_order(cfg, old, {"x"}, random.Random(1))]   # same day, same plan


def test_titles_hashtags_names_and_slots():
    import datetime as dt
    from joshiclip import upload
    known = ["Piyush", "Sourav", "Sahil", "Kunali", "Nana", "Nani"]
    assert upload.fix_names("Pyuish Phone Lene Aaiga 😂", known) == "Piyush Phone Lene Aaiga 😂"
    assert upload.fix_names("Sorav Aur Kunal Ki Nana", known) == "Sourav Aur Kunal Ki Nana"
    d = {"schedule": True, "publish_latest": ["08:30", "09:00"], "publish_old": ["07:00", "07:30", "18:00", "19:00"]}
    now = dt.datetime(2026, 11, 22, 2, 0, tzinfo=dt.timezone(dt.timedelta(minutes=330)))
    assert daily._slot(d, 0, 0, now, True) == "2026-11-22T03:00:00Z"     # 08:30 IST
    assert daily._slot(d, 0, 3, now, False) == "2026-11-22T13:30:00Z"    # 19:00 IST
