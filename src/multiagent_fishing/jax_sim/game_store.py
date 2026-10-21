"""Save, load, and browse full per-frame simulation snapshots."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import jax
import numpy as np

# Import your explicit map container and channel layout trackers
from multiagent_fishing.jax_sim.maps import JaxMap, JaxLayers
from multiagent_fishing.jax_sim.env import EnvConfig

FORMAT_VERSION = 2
SEP = "__"   # separates group from field name in the .npz keys


def _field_name(path: tuple) -> str:
    parts = []
    for k in path:
        name = getattr(k, "name", None)            # NamedTuple field
        if name is None:
            name = getattr(k, "key", getattr(k, "idx", k))  # dict key / list index
        parts.append(str(name))
    return ".".join(parts)


def flatten_named(tree: Any) -> Dict[str, np.ndarray]:
    """Pytree -> {"poachers.x": array, ...}. Any new field in the state
    shows up here with no code changes."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_field_name(path): leaf for path, leaf in leaves}


@dataclass(frozen=True)
class Game:
    index: int
    env: int
    start: int          # first frame, inclusive
    end: int            # last frame, inclusive
    truncated: bool     # True if cut off by the horizon
    summary: dict


class GameStore:
    """
    frames["before"][name]: (T, E, ...)  state at the start of each step
    frames["after"][name]:  (T, E, ...)  state after each step, before any reset
    extra[name]:            (T, E, ...)  actions, rewards
    done:                   (T, E) bool
    terrain:                {"height": (H, W), "biomass": (H, W)}
    """

    def __init__(self, frames: dict, extra: dict, done: np.ndarray, terrain: dict, meta: dict):
        self.frames = frames
        self.extra = extra
        self.done = np.asarray(done, dtype=bool)
        self.terrain = terrain
        self.meta = meta
        self.games = self._find_games()

    # ------------------------------------------------------------ building

    @classmethod
    def from_rollout(cls, traj: dict, map_data: JaxMap, env_cfg: EnvConfig, meta: dict = None, max_envs: int = None) -> GameStore:
        """Copy only the state, actions, rewards, and done flags to host.
        Observations and log-probs are skipped; they can be recomputed from
        the saved state if needed."""
        E = traj["done"].shape[1]
        keep = E if max_envs is None else min(max_envs, E)

        pick = {
            "before": traj["state"],
            "after": traj["next_state"],
            "extra": {"actions": traj["actions"], "rewards": traj["rewards"]},
            "done": traj["done"],
        }
        pick = jax.tree_util.tree_map(lambda a: a[:, :keep], pick)
        host = jax.device_get(pick)

        full_meta = {
            "format_version": FORMAT_VERSION,
            "env_config": dict(env_cfg._asdict()),
            # Fixed AttributeErrors by defaulting origin values to 0.0 and updating resolution hooks
            "grid": {"x0": 0.0, "y0": 0.0, "cell_km": float(map_data.resolution_km)},
            "horizon": int(host["done"].shape[0]),
            "num_envs": int(keep),
            **(meta or {}),
        }
        
        # Isolate the height map and biomass channel arrays back out of your compiled layers matrix
        height_array = np.asarray(map_data.layers[..., JaxLayers.HEIGHT])
        biomass_array = np.asarray(map_data.layers[..., JaxLayers.BIOMASS])

        return cls(
            frames={"before": flatten_named(host["before"]),
                    "after": flatten_named(host["after"])},
            extra=flatten_named(host["extra"]),
            done=host["done"],
            terrain={"height": height_array, "biomass": biomass_array},
            meta=full_meta,
        )

    # ------------------------------------------------------------- storage

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("path must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {"done": self.done}
        for group, d in (("before", self.frames["before"]),
                         ("after", self.frames["after"]),
                         ("extra", self.extra),
                         ("terrain", self.terrain)):
            for name, v in d.items():
                arrays[f"{group}{SEP}{name}"] = np.asarray(v)
        arrays["meta_json"] = np.array(json.dumps(self.meta))

        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> GameStore:
        frames = {"before": {}, "after": {}}
        extra, terrain = {}, {}
        groups = {"before": frames["before"], "after": frames["after"],
                  "extra": extra, "terrain": terrain}

        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(str(f["meta_json"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported format_version {meta.get('format_version')}")
            done = f["done"]
            for key in f.files:
                if key in ("done", "meta_json"):
                    continue
                group, name = key.split(SEP, 1)
                groups[group][name] = f[key]
        return cls(frames, extra, done, terrain, meta)

    # ------------------------------------------------------------- games

    def _find_games(self) -> List[Game]:
        T, E = self.done.shape
        games = []
        for e in range(E):
            done_ts = np.flatnonzero(self.done[:, e])
            starts = np.r_[0, done_ts + 1]
            ends = np.r_[done_ts, T - 1]
            truncated = np.r_[np.zeros(len(done_ts), bool), True]
            for s, t, trunc in zip(starts, ends, truncated):
                if s > t:      # game ended on the very last frame
                    continue
                games.append(Game(
                    index=len(games), env=int(e), start=int(s), end=int(t),
                    truncated=bool(trunc), summary=self._summary(e, s, t)))
        return games

    def _summary(self, e: int, s: int, t: int) -> dict:
        last = self.frames["after"]          # true state at the end of the game
        return {
            "steps": int(t - s + 1),
            "fish": float(last["poachers.cargo"][t, e].sum()),
            "catches": int(last["poachers.caught"][t, e].sum()),
            "survivors": int(last["poachers.active"][t, e].sum()),
        }

    def fields(self, group: str = "before") -> List[str]:
        """Names of every saved state field, e.g. 'poachers.heading'."""
        return sorted(self.frames[group])

    def table(self) -> List[dict]:
        return [{"index": g.index, "env": g.env, "start": g.start,
                 "end": g.end, "truncated": g.truncated, **g.summary}
                for g in self.games]

    def select(self, metric: str = "fish", how: str = "median", n: int = 1,
               include_truncated: bool = True, seed: int = 0) -> List[Game]:
        """metric: "steps" | "fish" | "catches" | "survivors"
        how:    "median" | "max" | "min" | "random" """
        pool = [g for g in self.games if include_truncated or not g.truncated]
        if not pool:
            raise ValueError("no games to select from")
        vals = np.array([g.summary[metric] for g in pool], dtype=float)

        if how == "median":
            order = np.argsort(np.abs(vals - np.median(vals)), kind="stable")
        elif how == "max":
            order = np.argsort(-vals, kind="stable")
        elif how == "min":
            order = np.argsort(vals, kind="stable")
        elif how == "random":
            order = np.random.default_rng(seed).permutation(len(pool))
        else:
            raise ValueError(f"unknown how={how!r}")
        return [pool[i] for i in order[:n]]

    def game_data(self, game: Game) -> dict:
        """Whole game as arrays, shape (steps, ...):
        frames: state at the start of each step
        final:  state after the last step
        extra:  actions and rewards"""
        sl, e = slice(game.start, game.end + 1), game.env
        return {
            "frames": {k: v[sl, e] for k, v in self.frames["before"].items()},
            "final": {k: v[game.end, e] for k, v in self.frames["after"].items()},
            "extra": {k: v[sl, e] for k, v in self.extra.items()},
        }

    def frame(self, game: Game, t: int) -> dict:
        """Single frame, t relative to the game start."""
        i, e = game.start + t, game.env
        return {k: v[i, e] for k, v in self.frames["before"].items()}
