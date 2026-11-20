import numpy as np

from joshiclip import viewers


C = [{"text": "6:15 oh my good😅😂 shisha tut gaya", "likes": 195},
     {"text": "6:12 popat hogaya 😂😂", "likes": 31},
     {"text": "6:13 Best part 😂😂", "likes": 24},
     {"text": "3:44 jisne ye padh liya uski mummy ke kasam like nhi kiya to", "likes": 3300},
     {"text": "0:04 first comment", "likes": 50},
     {"text": "2:40 piyush ne nasha ki hai", "likes": 40},
     {"text": "1:00 2:00 3:00 4:00", "likes": 0}]


def _A(hs):
    return {"_viewer_hits": hs, "comments_per_sec": viewers.density(hs, 900), "engine": {}}


def test_hits_drop_spam_and_split_lists():
    hs = viewers.hits(C, 900)
    ts = sorted(h["t"] for h in hs)
    assert 224.0 not in ts and 4.0 not in ts                     # like-bait + intro spam
    assert {372.0, 373.0, 375.0, 160.0} <= set(ts)
    lst = [h for h in hs if h["n_ts"] == 4]
    assert len(lst) == 4 and all(h["w"] == 0.25 for h in lst)    # a list = 4 weak votes


def test_score_needs_the_moment_inside_the_clip():
    A = _A(viewers.hits(C, 900))
    assert viewers.score(A, 340, 390) > 0.9
    assert viewers.score(A, 377, 420) == 0.0                     # starts right after the incident


def test_title_is_clean_safe_and_unique():
    A = _A(viewers.hits(C, 900))
    t = viewers.title(A, 340, 390, set(), ["Piyush"])
    assert t == "Oh My God Shisha Tut Gaya 😅😂"
    assert viewers.title(A, 340, 390, {t}, ["Piyush"]) == "Popat Ho Gaya 😂"   # "Best part" is generic
    assert viewers.title(A, 140, 190, set(), ["Piyush"]) is None              # unsafe line never becomes a title


def test_hotspot_from_cluster():
    hs = viewers.hits(C * 2, 900)                                # two copies -> enough distinct-ish weight
    A = _A([dict(h, text=h["text"] + str(i)) for i, h in enumerate(hs)])
    spots = viewers.hotspots(A, {"engine": {"must_peak_strength": 3.0}, "viewers": {}})
    assert spots and 370 <= spots[0]["t"] <= 378 and spots[0]["strength"] >= 3.0
