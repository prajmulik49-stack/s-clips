"""Load config.yaml and hash config sections for cache keys."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config(dict):
    """dict with path helpers. `cfg.path("work")` resolves relative to the project root."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p

    def section_hash(self, *keys: str) -> str:
        return stable_hash({k: self.get(k) for k in keys})


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def load(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(_merge(data, overrides or {}))


def load_lexicon(cfg: Config) -> dict:
    p = cfg.path("lexicon")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
