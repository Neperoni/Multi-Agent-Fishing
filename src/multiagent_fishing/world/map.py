import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

KM_PER_DEGREE = 111.32
SEA_LEVEL = 0.0  # height map is depth/height, so land is height >= 0

class Layers(IntEnum):
    """Layout of Map.layers, shape (rows, cols, len(Channel)), float32. Used as array indices."""
    HEIGHT = 0              # meters, >= 0 is land
    BIOMASS = 1             # density in [0, 1]
    SHORE_DIST = 2          # km to nearest land, 0 on land
    NEAREST_PORT_ID = 3     # index into Map.ports, -1 if no port is reachable
    NEAREST_PORT_DIST = 4   # km along water to that port
    NEAREST_PORT_DIR_X = 5  # unit vector toward that port (world x, or boat-forward)
    NEAREST_PORT_DIR_Y = 6  # world y, or boat-right


OOB_HEIGHT = 1.0       # out-of-bounds points look like land
RAY_STEP_CELLS = 0.5   # dense ray step, in cells, so thin land can't be skipped


class RayHit(NamedTuple):
    """
    Where a ray stopped. Each field is a scalar for a single ray and an array for a batch.
    `hit` is True if the ray reached land or the map edge within range; otherwise x, y and
    distance describe the end of the ray at max range.
    """
    x_km: float | np.ndarray
    y_km: float | np.ndarray
    distance_km: float | np.ndarray
    hit: bool | np.ndarray


class RayResult(NamedTuple):
    hit: RayHit
    samples: np.ndarray  # channel values at evenly spaced points along the ray: (..., n_samples, C)


def _as_batch(x) -> tuple[np.ndarray, bool]:
    """Accepts one (2,) point or an (N, 2) batch. Returns (N, 2) and whether input was single."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a[None, :], True
    return a, False


def _to_ego(dx, dy, heading):
    """
    Rotates world vectors into the boat frame: (forward, right).
    Heading is radians with +x = cos, +y = sin, so right is (sin h, -cos h).
    """
    c, s = np.cos(heading), np.sin(heading)
    return dx * c + dy * s, dx * s - dy * c


def _rotate_dir_channels(arr: np.ndarray, heading: np.ndarray) -> None:
    """In place: rewrite the port direction channels of an (M, C) array into the boat frame."""
    fwd, right = _to_ego(
        arr[:, Layers.NEAREST_PORT_DIR_X].astype(np.float64),
        arr[:, Layers.NEAREST_PORT_DIR_Y].astype(np.float64),
        heading,
    )
    arr[:, Layers.NEAREST_PORT_DIR_X] = fwd
    arr[:, Layers.NEAREST_PORT_DIR_Y] = right


@dataclass(eq=False)
class Map:
    """
    Finished, static map. Everything derived (shore distance, port distance and flow fields)
    is computed by the generator and stored here, so loading a saved map does no heavy work.
    Positions are (x, y) in km, origin at the south-west corner, y increasing north.
    All query methods accept one point or an (N, 2) batch.
    """
    layers: np.ndarray       # (rows, cols, len(Channel)) float32, see channel constants
    port_dist: np.ndarray    # (n_ports, rows, cols) km along water, capped at unreachable_km
    port_flow: np.ndarray    # (n_ports, 2, rows, cols) unit vectors (x, y) toward each port
    width_km: float
    height_km: float
    resolution_km: float     # cell size
    ports: list[dict]        # port id is the list index
    origin_lon: float = 0.0  # lon/lat of the map's (0, 0) corner, for converting back to degrees
    origin_lat: float = 0.0
    lon_scale: float = 1.0   # cos(center latitude) used in the projection

    rows: int = field(init=False)
    cols: int = field(init=False)
    land_mask: np.ndarray = field(init=False, repr=False)
    _oob_row: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.layers.ndim != 3 or self.layers.shape[-1] != len(Layers):
            raise ValueError(f"layers must have shape (rows, cols, {len(Layers)})")
        self.layers.setflags(write=False)
        self.rows, self.cols = self.layers.shape[:2]
        self.land_mask = self.layers[..., Layers.HEIGHT] >= SEA_LEVEL

        oob = np.zeros(len(Layers), dtype=np.float32)
        oob[Layers.HEIGHT] = OOB_HEIGHT
        oob[Layers.NEAREST_PORT_ID] = -1.0
        oob[Layers.NEAREST_PORT_DIST] = self.unreachable_km
        self._oob_row = oob

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def unreachable_km(self) -> float:
        """Value used for 'no path': the map diagonal. Keeps distances finite for the network."""
        return math.hypot(self.width_km, self.height_km)

    def is_in_bounds(self, x_km: float, y_km: float) -> bool:
        return 0.0 <= x_km <= self.width_km and 0.0 <= y_km <= self.height_km

    def km_to_lonlat(self, x_km, y_km):
        lon = self.origin_lon + np.asarray(x_km) / (KM_PER_DEGREE * self.lon_scale)
        lat = self.origin_lat + np.asarray(y_km) / KM_PER_DEGREE
        return lon, lat

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cells(self, x: np.ndarray, y: np.ndarray):
        """Cell indices and in-bounds mask for arrays of km coordinates (any shape)."""
        inb = (x >= 0) & (x <= self.width_km) & (y >= 0) & (y <= self.height_km)
        cols = np.clip(np.floor(x / self.resolution_km).astype(np.intp), 0, self.cols - 1)
        rows = np.clip(np.floor(y / self.resolution_km).astype(np.intp), 0, self.rows - 1)
        return rows, cols, inb

    def _gather(self, pts: np.ndarray) -> np.ndarray:
        """(M, 2) km points -> (M, C) channel values, out-of-bounds points get the fill row."""
        rows, cols, inb = self._cells(pts[:, 0], pts[:, 1])
        out = self.layers[rows, cols]  # fancy indexing returns a writable copy
        out[~inb] = self._oob_row
        return out

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_point(self, positions, heading=None) -> np.ndarray:
        """
        All channels at each position: (N, C), or (C,) for a single position.
        Out-of-bounds positions return land-like values. With `heading` (scalar or per-position,
        radians) the port direction channels are returned as (forward, right) instead of (x, y).
        """
        p, single = _as_batch(positions)
        out = self._gather(p)
        if heading is not None:
            _rotate_dir_channels(out, np.broadcast_to(np.asarray(heading, dtype=np.float64), (len(p),)))
        return out[0] if single else out

    def sample_vector(
        self,
        origins,
        headings,
        max_range_km: float,
        n_samples: int = 0,
        frame_headings=None,
    ) -> RayResult:
        """
        Casts one ray per origin along `headings` (radians).

        Hit detection samples the land mask every half cell, so thin land can't be skipped.
        `hit` is a RayHit for the first land cell or map edge along the ray (accurate to the
        step); fields are arrays of length N for a batch. `samples` holds all channels at
        `n_samples` evenly spaced points out to max range (not truncated at the hit, so check
        `hit` before trusting samples past it). If `frame_headings` is given, the port
        direction channels in the samples are rotated into that frame instead of world x/y.
        """
        if max_range_km <= 0:
            raise ValueError("max_range_km must be positive")

        o, single = _as_batch(origins)
        n = len(o)
        h = np.broadcast_to(np.asarray(headings, dtype=np.float64), (n,))
        dirs = np.stack([np.cos(h), np.sin(h)], axis=1)  # (N, 2)

        # Dense pass: first blocked step (land or off the map)
        step = RAY_STEP_CELLS * self.resolution_km
        ts = np.linspace(0.0, max_range_km, int(math.ceil(max_range_km / step)) + 1)
        pts = o[:, None, :] + ts[None, :, None] * dirs[:, None, :]  # (N, T, 2)
        rows, cols, inb = self._cells(pts[..., 0], pts[..., 1])
        blocked = ~inb | self.land_mask[rows, cols]
        hit = blocked.any(axis=1)
        first = blocked.argmax(axis=1)
        distance = np.where(hit, ts[first], max_range_km)
        ray = RayHit(o[:, 0] + distance * dirs[:, 0], o[:, 1] + distance * dirs[:, 1], distance, hit)

        # Sparse pass: full channel samples along the ray
        if n_samples > 0:
            ts_s = max_range_km * np.arange(1, n_samples + 1) / n_samples
            pts_s = (o[:, None, :] + ts_s[None, :, None] * dirs[:, None, :]).reshape(-1, 2)
            samples = self._gather(pts_s)
            if frame_headings is not None:
                fh = np.broadcast_to(np.asarray(frame_headings, dtype=np.float64), (n,))
                _rotate_dir_channels(samples, np.repeat(fh, n_samples))
            samples = samples.reshape(n, n_samples, len(Layers))
        else:
            samples = np.zeros((n, 0, len(Layers)), dtype=np.float32)

        if single:
            return RayResult(RayHit(*(f[0].item() for f in ray)), samples[0])
        return RayResult(ray, samples)

    def sample_fan(
        self,
        positions,
        headings,
        rel_angles,
        max_range_km: float,
        n_samples: int = 0,
    ) -> RayResult:
        """
        sample_vector for a fan of rays at angles relative to each boat's heading.
        Shapes: each RayHit field is (N, A) and samples is (N, A, n_samples, C); the leading N
        is dropped for a single position. Port direction channels are in the boat frame.
        """
        p, single = _as_batch(positions)
        n = len(p)
        h = np.broadcast_to(np.asarray(headings, dtype=np.float64), (n,))
        a = np.asarray(rel_angles, dtype=np.float64)
        n_rays = len(a)

        r = self.sample_vector(
            np.repeat(p, n_rays, axis=0),
            (h[:, None] + a[None, :]).reshape(-1),
            max_range_km,
            n_samples,
            frame_headings=np.repeat(h, n_rays),
        )
        hit = RayHit(*(f.reshape(n, n_rays) for f in r.hit))
        samples = r.samples.reshape(n, n_rays, n_samples, len(Layers))
        if single:
            return RayResult(RayHit(*(f[0] for f in hit)), samples[0])
        return RayResult(hit, samples)

    def raycast_to_shore(self, positions, headings, max_range_km: float):
        """RayHit to the first land or map edge along each heading."""
        return self.sample_vector(positions, headings, max_range_km, n_samples=0).hit

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------
    def nav_to_port(self, port_id: int, positions, heading=None) -> np.ndarray:
        """
        [path_distance_km, dir_x, dir_y] toward one port along water: (N, 3), or (3,) for a
        single position. Distance is `unreachable_km` and direction (0, 0) if there is no path
        or the position is out of bounds. With `heading` the direction is (forward, right).
        """
        p, single = _as_batch(positions)
        rows, cols, inb = self._cells(p[:, 0], p[:, 1])
        
        dist = np.where(inb, self.port_dist[port_id][rows, cols], self.unreachable_km)
        flow = self.port_flow[port_id][:, rows, cols].astype(np.float64)
        
        dx = np.where(inb, flow[0], 0.0)
        dy = np.where(inb, flow[1], 0.0)
        
        if heading is not None:
            hd = np.broadcast_to(np.asarray(heading, dtype=np.float64), (len(p),))
            dx, dy = _to_ego(dx, dy, hd)
            
        out = np.stack([dist, dx, dy], axis=1).astype(np.float32)
        return out[0] if single else out

    def nearest_port(self, positions, heading=None) -> np.ndarray:
        """[port_id, path_distance_km, dir_x, dir_y] for the nearest port by water path."""
        return self.sample_point(positions, heading)[..., Layers.NEAREST_PORT_ID:]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        meta = {
            "width_km": self.width_km,
            "height_km": self.height_km,
            "resolution_km": self.resolution_km,
            "ports": self.ports,
            "origin_lon": self.origin_lon,
            "origin_lat": self.origin_lat,
            "lon_scale": self.lon_scale,
        }
        np.savez_compressed(
            path,
            layers=self.layers,
            port_dist=self.port_dist,
            port_flow=self.port_flow,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Map":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            return cls(
                layers=data["layers"],
                port_dist=data["port_dist"],
                port_flow=data["port_flow"],
                **meta,
            )