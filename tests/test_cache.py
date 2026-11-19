import numpy as np

from joshiclip.cache import StageCache


def test_cache_reuse_and_invalidation(tmp_path):
    c = StageCache(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"x": 1}, {"a": np.arange(3)}

    d, a = c.run("s", 1, {"t": 0.5}, fn, log=lambda *_: None)
    d2, a2 = c.run("s", 1, {"t": 0.5}, fn, log=lambda *_: None)
    assert len(calls) == 1 and d2 == {"x": 1} and list(a2["a"]) == [0, 1, 2]
    c.run("s", 1, {"t": 0.6}, fn, log=lambda *_: None)     # config change -> rerun
    c.run("s", 2, {"t": 0.6}, fn, log=lambda *_: None)     # code version change -> rerun
    assert len(calls) == 3
