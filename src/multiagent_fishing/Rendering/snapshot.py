"""
GameSnapshot: the read-only contract between a game and anything that wants to look at it.

The game library produces these; views (and trace recorders/replayers) consume them.
No pygame, no file paths, no clock in here.

Coordinates are in km on the flattened map grid: x east, y north, origin at the
south-west corner. Headings are radians, 0 = north, increasing clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence

import numpy as np


class Role(IntEnum):
    """What a boat *is* to the game. How it looks is the view's business."""
    FISHING = 0
    PATROL = 1


@dataclass
class GameSnapshot:
    tick: int
    x_km: np.ndarray                    # (N,)
    y_km: np.ndarray                    # (N,)
    heading: np.ndarray                 # (N,) radians
    role: np.ndarray                    # (N,) ints from Role

    game_id: int = 0                    # which game this came from (views key caches on it)
    names: Optional[Sequence[str]] = None   # length N; the view invents F1, P2... if None
    active: Optional[np.ndarray] = None     # (N,) bool; False = not drawn (padded / retired boats)

    # Per-game fish/biomass field, shape (rows, cols) matching the map grid, values in [0, 1],
    # row 0 = south. BUMP fish_version WHENEVER THE FIELD CHANGES: views only rebuild their
    # overlay image when (game_id, fish_version) changes.
    fish: Optional[np.ndarray] = None
    fish_version: int = 0

    # Free-form HUD lines, e.g. {"score": 12.5, "fish caught": 31}
    info: dict = field(default_factory=dict)
