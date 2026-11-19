"""Steps 2-3 logic on a synthetic analysis: lexicon matching, sentences, filters, scoring determinism, selection."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from joshiclip import candidates, config, pipeline, selection
from joshiclip.analysis import speech
from joshiclip.text import lex


@pytest.fixture(scope="module")
def cfg():
    c = config.load()
    lex.set_lexicon(config.load_lexicon(c))
    return c


def test_phonetic_matching(cfg):
    assert lex.phrase_in("welcome to my vlog", "गाइस वेलकम टू माय न्यू व्लॉग")
    assert lex.phrase_in("namaste doston", "नमस्ते दोस्तों कैसे हो")
    assert lex.phrase_in("bell icon", "बेल आइकन दबा दो")
    assert not lex.phrase_in("kya", "इसके के की बाल")
    assert lex.first_word_dependent("और ये देखो") == "aur"
    assert lex.first_word_dependent("तो फिर") in ("toh", "to")
    assert lex.first_word_dependent("देखो ये क्या है") is None


def _w(t, s, e, p=0.9):
    return {"w": t, "s": s, "e": e, "p": p}


def test_sentences(cfg):
    words = [_w("देखो", 0.0, 0.3), _w("ये", 0.35, 0.5), _w("क्या", 0.55, 0.8), _w("है?", 0.85, 1.1),
             _w("मैं", 2.0, 2.2), _w("जा", 2.25, 2.4), _w("रहा", 2.45, 2.6), _w("हूं", 2.65, 2.9),
             _w("और", 4.0, 4.2)]
    s = speech.build_sentences(words, cfg["sentences"])
    assert [x["text"] for x in s] == ["देखो ये क्या है?", "मैं जा रहा हूं", "और"]
    assert s[0]["complete"] and s[0]["punct_end"]
    assert s[1]["complete"]            # ends at a 1.1 s pause, not on a connective
    assert not s[2]["complete"]        # dangling "aur"


def _synthetic(n_sent=60, sent_len=4.0, gap=0.8):
    rng = np.random.default_rng(0)
    vocab = ["देखो", "ओरियो", "कुत्ता", "पापा", "मम्मी", "खाना", "गाड़ी", "बाहर", "घर", "मजे", "बहुत", "अच्छा"]
    words, sents = [], []
    t = 0.0
    for i in range(n_sent):
        ws = []
        for k in range(4):
            w = {"w": vocab[(i * 3 + k) % len(vocab)] + ("।" if k == 3 else ""), "s": round(t, 3), "e": round(t + 0.8, 3), "p": 0.9}
            ws.append(w); t += 1.0
        words += ws
        t += gap
    for i, w in enumerate(words):
        w["i"] = i
    sents = speech.build_sentences(words, config.load()["sentences"])
    for k, s in enumerate(sents):
        s.update({"spk": k % 2, "is_boundary": k % 10 == 0, "topic": k // 10, "boundary_score": 0.5, "location_score": 0.0})
    n = int(np.ceil(t)) + 1
    ps = {k: np.zeros(n, np.float32) for k in ("rms_max", "pitch_std", "laugh", "cheer", "scream", "music", "gasp", "cry",
                                                 "dog", "motion", "location", "scene_cut", "words")}
    ps["rms"] = (-30 + rng.normal(0, 2, n)).astype(np.float32)
    ps["pitch_med"] = np.zeros(n, np.float32)
    ps["speech"] = np.ones(n, np.float32)
    ps["faces"] = np.ones(n, np.float32)
    ps["ambience_vec"] = np.zeros((n, 8), np.float32)
    for pt in (60, 150, 240):
        ps["laugh"][pt:pt + 3] = 0.8
    peaks = [{"id": i, "t": float(t0 + 1), "strength": 5.0 - i * 0.1, "type": "laugh", "signals": {}} for i, t0 in enumerate((60, 150, 240))]
    return {"vid": "synthetic", "duration": float(n - 1), "n_sec": n, "fps": 30, "words": words, "sentences": sents,
            "per_sec": ps, "peaks": peaks, "topics": {"topic_keys": {}, "is_boundary": [s["is_boundary"] for s in sents]},
            "scene_cuts": [], "lexicon": lex.lexicon()}


def test_plan_deterministic_and_rules(cfg):
    A = _synthetic()
    c1, s1 = pipeline.plan(cfg, copy.deepcopy(A))
    c2, s2 = pipeline.plan(cfg, copy.deepcopy(A))
    assert [c["final"] for c in c1] == [c["final"] for c in c2] and s1["winners"] == s2["winners"]
    for c in c1:
        assert cfg["candidates"]["len_allow_s"][0] <= c["dur"] <= cfg["candidates"]["len_allow_s"][1] + 0.5
        s0 = A["sentences"][c["si"]]
        if not s0["is_boundary"] and s0["pause_before"] < cfg["filters"]["min_pause_before_s"]:
            assert c["reject"]
    win = [c1[i] for i in s1["winners"]]
    assert 1 <= len(win) <= cfg["selection"]["max_shorts"]
    for i, a in enumerate(win):
        for b in win[i + 1:]:
            assert selection.overlap(a, b) <= cfg["selection"]["max_overlap_frac"]
    assert len({c["cluster"] for c in win}) == len(win)       # one Short per moment cluster
    assert all(c["title"] and len(c["title"]) <= 40 for c in win)


def test_intro_phrase_rejected(cfg):
    A = _synthetic()
    s = A["sentences"][10]
    w0 = A["words"][s["w0"]]
    for k, t in enumerate(["वेलकम", "टू", "माय", "व्लॉग"]):
        A["words"][s["w0"] + k]["w"] = t
    s["text"] = " ".join(A["words"][i]["w"] for i in range(s["w0"], s["w1"]))
    cands = candidates.generate(A, cfg)
    candidates.hard_filters(A, cands, cfg)
    hit = [c for c in cands if c["si"] == 10]
    assert hit and all(c["reject_all"] and any(r.startswith("intro_outro_phrase") for r in c["reject_all"]) for c in hit)
    assert w0["s"] == s["s"]


def test_boundaries_clean_cuts_and_deterministic(cfg):
    from joshiclip import boundaries
    A = _synthetic()
    n = A["n_sec"]
    rms10 = np.full(n * 100, -30.0, np.float32)
    for w in A["words"]:                              # speech is louder than the gaps
        rms10[int(w["s"] * 100):int(w["e"] * 100)] = -12.0
    A["rms10"] = rms10
    A["duration"] = float(n - 1)
    b1 = boundaries.best_span(A, 150.0, cfg)
    b2 = boundaries.best_span(A, 150.0, cfg)
    assert b1 == b2 and b1 is not None
    c = cfg["boundaries"]
    assert c["min_s"] <= b1["dur"] <= c["max_s"]
    assert b1["start"] < 150.0 < b1["end"]
    for w in A["words"]:                              # no cut inside a word
        assert not (w["s"] < b1["start"] < w["e"]) and not (w["s"] < b1["end"] < w["e"])
