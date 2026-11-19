"""Orchestration: ingest -> analyze (cached) -> candidates -> filters -> scoring -> selection -> render -> checks."""
from __future__ import annotations

from pathlib import Path

from .config import Config


def workdir_for(cfg: Config, vid: str) -> Path:
    return cfg.path("work") / vid
