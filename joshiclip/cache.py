"""Stage cache: work/<video_id>/<stage>.json (+ optional <stage>.npz) with a dependency key.

A stage is reused only when its key (stage version + config sections + upstream keys) matches,
so changing one threshold reruns only the stages downstream of it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import stable_hash


class StageCache:
    def __init__(self, workdir: Path):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}
        self.keys: dict[str, str] = {}

    def key(self, stage: str, version: int, deps: dict) -> str:
        k = stable_hash({"stage": stage, "v": version, "deps": deps})
        self.keys[stage] = k
        return k

    def _meta(self, stage):
        return self.dir / f"{stage}.meta.json"

    def get(self, stage: str, key: str):
        m = self._meta(stage)
        if not m.exists():
            return None
        meta = json.loads(m.read_text(encoding="utf-8"))
        if meta.get("key") != key:
            return None
        self.timings[stage] = meta.get("seconds", 0.0)
        data = json.loads((self.dir / f"{stage}.json").read_text(encoding="utf-8"))
        npz = self.dir / f"{stage}.npz"
        arrays = dict(np.load(npz, allow_pickle=False)) if npz.exists() else {}
        return data, arrays

    def put(self, stage: str, key: str, data, arrays: dict | None = None, seconds: float = 0.0):
        (self.dir / f"{stage}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        if arrays:
            np.savez(self.dir / f"{stage}.npz", **arrays)
        self._meta(stage).write_text(json.dumps({"key": key, "seconds": round(seconds, 2)}), encoding="utf-8")
        self.timings[stage] = seconds

    def run(self, stage: str, version: int, deps: dict, fn, log=print):
        """Return cached (data, arrays) or compute fn() -> (data, arrays) and store it."""
        key = self.key(stage, version, deps)
        hit = self.get(stage, key)
        if hit is not None:
            log(f"  [cache] {stage}")
            return hit
        log(f"  [run]   {stage} ...")
        t = time.perf_counter()
        data, arrays = fn()
        dt = time.perf_counter() - t
        self.put(stage, key, data, arrays, dt)
        log(f"  [done]  {stage} in {dt:.1f}s")
        return data, (arrays or {})
