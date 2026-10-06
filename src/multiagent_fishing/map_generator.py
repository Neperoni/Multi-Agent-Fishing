# multiagent_fishing/map_generator.py

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

KM_PER_DEGREE = 111.32

@dataclass(frozen=True)
class RayHit:
    distance_km: float
    x_km: float
    y_km: float
    height: float  # nan when the hit is the map edge
    is_bounds: bool = False


class Map:
    """
    Represents a localized grid containing underlying height data, generated biological/biomass
    densities, and structural simulation ports. Handles spatial calculations in kilometers.

    Terrain-derived fields (land mask, shore distance, port flow fields) are built once at
    construction. If _height_grid is modified afterwards they must be rebuilt.
    """

    def __init__(self, height_grid: np.ndarray, biomass_grid: np.ndarray, width_km: float, height_km: float, ports: list[dict]):
        self._height_grid = height_grid
        self._biomass_grid = biomass_grid
        self.width_km = width_km
        self.height_km = height_km
        self.rows, self.cols = height_grid.shape
        self._ports = ports

        # NaN >= SEA_LEVEL is False, so NaN cells count as water
        self._land_mask = height_grid >= 0
        self._build_shore_distance()
        self._build_flow_fields()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def ports(self) -> list[dict]:
        return self._ports

    @property
    def shape(self):
        return self._height_grid.shape

    @property
    def cols_count(self) -> int:
        return self.cols

    @property
    def rows_count(self) -> int:
        return self.rows

    @property
    def km_per_pixel(self) -> float:
        if self.cols > 1:
            return self.width_km / self.cols
        return 0.0

    @property
    def height_data(self) -> np.ndarray:
        return self._height_grid

    @property
    def biomass_data(self) -> np.ndarray:
        return self._biomass_grid

    @property
    def shore_distance_data(self) -> np.ndarray:
        return self._shore_dist_km

    # ------------------------------------------------------------------
    # Basic sampling
    # ------------------------------------------------------------------
    def is_in_bounds(self, x_km: float, y_km: float) -> bool:
        return 0.0 <= x_km <= self.width_km and 0.0 <= y_km <= self.height_km

    def get_height_at_pixel(self, x_km: float, y_km: float, method: str = "closest") -> float:
        return self._sample_grid(self._height_grid, x_km, y_km, method)

    def get_biomass_at_pixel(self, x_km: float, y_km: float, method: str = "closest") -> float:
        return self._sample_grid(self._biomass_grid, x_km, y_km, method)

    def km_to_pixel_coords(self, x_km: float, y_km: float) -> tuple[float, float]:
        res = self.km_per_pixel
        if res <= 0.0:
            return 0.0, 0.0
        return x_km / res, y_km / res

    def _sample_grid(self, grid: np.ndarray, x_km: float, y_km: float, method: str) -> float:
        if not self.is_in_bounds(x_km, y_km):
            raise IndexError(f"Kilometer coordinates ({x_km:.2f}, {y_km:.2f}) out of bounds")

        px, py = self.km_to_pixel_coords(x_km, y_km)

        if method == "closest":
            col = min(int(np.floor(px)), self.cols - 1)
            row = min(int(np.floor(py)), self.rows - 1)
            return float(grid[row, col])

        elif method == "bilinear":
            c0 = int(np.floor(px))
            r0 = int(np.floor(py))
            c1 = min(c0 + 1, self.cols - 1)
            r1 = min(r0 + 1, self.rows - 1)

            tx = px - c0
            ty = py - r0

            val_bottom = (1.0 - tx) * grid[r0, c0] + tx * grid[r0, c1]
            val_top = (1.0 - tx) * grid[r1, c0] + tx * grid[r1, c1]
            return float((1.0 - ty) * val_bottom + ty * val_top)
        else:
            raise ValueError(f"Unknown interpolation mode: '{method}'")

    def _cell_of(self, x_km: float, y_km: float) -> tuple[int, int]:
        """(row, col) of the cell containing a point, clamped to the grid."""
        res = self.km_per_pixel
        col = min(max(int(x_km / res), 0), self.cols - 1)
        row = min(max(int(y_km / res), 0), self.rows - 1)
        return row, col

    def sample_points(self, grid: np.ndarray, xs, ys, fill: float = 0.0) -> np.ndarray:
        """Non-raising, vectorized nearest-cell sampling. Out-of-bounds points get `fill`."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        res = self.km_per_pixel
        ok = (xs >= 0) & (xs <= self.width_km) & (ys >= 0) & (ys <= self.height_km)
        cols = np.clip(np.floor(xs / res).astype(int), 0, self.cols - 1)
        rows = np.clip(np.floor(ys / res).astype(int), 0, self.rows - 1)
        return np.where(ok, grid[rows, cols], fill)

    def sample_fan(self, grid, x_km, y_km, heading, rel_angles, ranges, fill: float = 0.0) -> np.ndarray:
        """Samples at egocentric offsets. Returns shape (len(rel_angles), len(ranges))."""
        ang = heading + np.asarray(rel_angles, dtype=float)[:, None]
        rng = np.asarray(ranges, dtype=float)[None, :]
        return self.sample_points(grid, x_km + rng * np.cos(ang), y_km + rng * np.sin(ang), fill)

    def biomass_fan(self, x_km, y_km, heading, rel_angles, ranges) -> np.ndarray:
        return self.sample_fan(self._biomass_grid, x_km, y_km, heading, rel_angles, ranges)

    # ------------------------------------------------------------------
    # Land / water queries and shore distance field
    # ------------------------------------------------------------------
    def _build_shore_distance(self) -> None:
        """Distance (km) from each water cell to the nearest land cell. 0 on land."""
        if self._land_mask.any():
            self._shore_dist_km = distance_transform_edt(~self._land_mask) * self.km_per_pixel
        else:
            self._shore_dist_km = np.full(self._land_mask.shape, np.inf)

    def is_water(self, x_km: float, y_km: float) -> bool:
        if not self.is_in_bounds(x_km, y_km):
            return False
        r, c = self._cell_of(x_km, y_km)
        return not self._land_mask[r, c]

    def shore_distance_at(self, x_km: float, y_km: float) -> float:
        """Distance to nearest land in km. 0 on land or out of bounds."""
        if not self.is_in_bounds(x_km, y_km):
            return 0.0
        r, c = self._cell_of(x_km, y_km)
        return float(self._shore_dist_km[r, c])

    # ------------------------------------------------------------------
    # Port flow fields
    # ------------------------------------------------------------------
    def _build_flow_fields(self) -> None:
        """One Dijkstra per port over water cells, plus a steepest-descent direction field."""
        res = self.km_per_pixel
        self._port_dist: list[np.ndarray] = []  # path distance to port in km, inf if unreachable
        self._port_flow: list[np.ndarray] = []  # (2, rows, cols) unit vectors (x, y) toward the port
        for port in self._ports:
            cost = np.where(self._land_mask, np.inf, 1.0)
            r, c = self._cell_of(port["grid_x"], port["grid_y"])
            cost[r, c] = 1.0  # guarantee the start cell is traversable
            cum, _ = MCP_Geometric(cost, fully_connected=True).find_costs([(r, c)])
            cum = cum * res
            self._port_dist.append(cum)
            self._port_flow.append(self._steepest_descent(cum))

    @staticmethod
    def _steepest_descent(cum: np.ndarray) -> np.ndarray:
        ny, nx = cum.shape
        pad = np.pad(cum, 1, constant_values=np.inf)
        best = cum.copy()
        fx = np.zeros_like(cum)
        fy = np.zeros_like(cum)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                nb = pad[1 + oy:1 + oy + ny, 1 + ox:1 + ox + nx]
                better = nb < best
                n = math.hypot(ox, oy)
                best = np.where(better, nb, best)
                fx = np.where(better, ox / n, fx)
                fy = np.where(better, oy / n, fy)
        return np.stack([fx, fy])

    def port_flow_at(self, port_idx: int, x_km: float, y_km: float) -> tuple[float, float, float]:
        """
        (dir_x, dir_y, path_dist_km) toward the port along water.
        Returns (0, 0, inf) if out of bounds or unreachable.
        The direction is in world coordinates, so the caller rotates it into the boat's frame.
        """
        if not self.is_in_bounds(x_km, y_km):
            return 0.0, 0.0, float("inf")
        r, c = self._cell_of(x_km, y_km)
        f = self._port_flow[port_idx]
        return float(f[0, r, c]), float(f[1, r, c]), float(self._port_dist[port_idx][r, c])

    # ------------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------------
    def raycast(
        self,
        x_km: float,
        y_km: float,
        dir_x: float,
        dir_y: float,
        max_dist_km: float,
        predicate: Callable[[float], bool],
        bounds_as_wall: bool = True,
    ) -> RayHit | None:
        """
        Grid traversal (Amanatides-Woo DDA) from (x_km, y_km) along (dir_x, dir_y).
        Returns the first cell whose height satisfies `predicate`. If the ray leaves the map
        within max_dist_km, returns a bounds hit when bounds_as_wall is True, else None.
        Visits every cell the ray passes through, so thin land features can't be skipped.
        """
        res = self.km_per_pixel
        norm = math.hypot(dir_x, dir_y)
        if res <= 0.0 or norm == 0.0 or max_dist_km <= 0.0 or not self.is_in_bounds(x_km, y_km):
            return None

        dx, dy = dir_x / norm, dir_y / norm
        px, py = x_km / res, y_km / res  # start in pixel units
        max_t = max_dist_km / res

        col = min(int(px), self.cols - 1)
        row = min(int(py), self.rows - 1)
        step_c = 1 if dx > 0 else -1
        step_r = 1 if dy > 0 else -1

        inf = float("inf")
        t_delta_x = abs(1.0 / dx) if dx != 0 else inf
        t_delta_y = abs(1.0 / dy) if dy != 0 else inf
        t_max_x = ((col + 1 - px) / dx if dx > 0 else (px - col) / -dx) if dx != 0 else inf
        t_max_y = ((row + 1 - py) / dy if dy > 0 else (py - row) / -dy) if dy != 0 else inf

        t = 0.0
        while t <= max_t:
            h = float(self._height_grid[row, col])
            if predicate(h):
                return RayHit(t * res, (px + dx * t) * res, (py + dy * t) * res, h)

            if t_max_x < t_max_y:
                t = t_max_x
                t_max_x += t_delta_x
                col += step_c
            else:
                t = t_max_y
                t_max_y += t_delta_y
                row += step_r

            if not (0 <= col < self.cols and 0 <= row < self.rows):
                if bounds_as_wall and t <= max_t:
                    return RayHit(t * res, (px + dx * t) * res, (py + dy * t) * res, float("nan"), True)
                return None
        return None

    def raycast_shore(
        self,
        x_km: float,
        y_km: float,
        dir_x: float,
        dir_y: float,
        max_dist_km: float,
        bounds_as_wall: bool = True,
    ) -> RayHit | None:
        """First land cell along the ray (exact DDA), or None."""
        return self.raycast(x_km, y_km, dir_x, dir_y, max_dist_km, lambda h: h >= 0, bounds_as_wall)

    def _distance_to_bounds(self, x: float, y: float, dx: float, dy: float) -> float:
        tx = (self.width_km - x) / dx if dx > 0 else (x / -dx if dx < 0 else float("inf"))
        ty = (self.height_km - y) / dy if dy > 0 else (y / -dy if dy < 0 else float("inf"))
        return min(tx, ty)

    def raycast_sphere(
        self,
        x_km: float,
        y_km: float,
        dir_x: float,
        dir_y: float,
        max_dist_km: float,
        bounds_as_wall: bool = True,
    ) -> RayHit | None:
        """
        Sphere tracing against the shore distance field: each step advances by the distance
        to the nearest land (minus one cell of slack), so open water is crossed in a few
        steps. Near land it falls back to half-cell steps. Hit distance is accurate to
        about half a cell.
        """
        res = self.km_per_pixel
        norm = math.hypot(dir_x, dir_y)
        if res <= 0.0 or norm == 0.0 or max_dist_km <= 0.0 or not self.is_in_bounds(x_km, y_km):
            return None
        dx, dy = dir_x / norm, dir_y / norm

        t_exit = self._distance_to_bounds(x_km, y_km, dx, dy)
        limit = min(max_dist_km, t_exit)
        min_step = 0.5 * res

        t = 0.0
        while True:
            px, py = x_km + dx * t, y_km + dy * t
            r, c = self._cell_of(px, py)
            if self._land_mask[r, c]:
                return RayHit(t, px, py, float(self._height_grid[r, c]))
            if t >= limit:
                break
            t = min(t + max(self._shore_dist_km[r, c] - res, min_step), limit)

        if bounds_as_wall and t_exit <= max_dist_km:
            return RayHit(t_exit, x_km + dx * t_exit, y_km + dy * t_exit, float("nan"), True)
        return None

    def raycast_fan(
        self,
        x_km: float,
        y_km: float,
        heading: float,
        rel_angles,
        max_dist_km: float,
        bounds_as_wall: bool = True,
    ) -> np.ndarray:
        """Normalized hit distances in [0, 1], one per angle. 1.0 means nothing within range."""
        out = np.ones(len(rel_angles))
        for i, a in enumerate(rel_angles):
            h = self.raycast_sphere(
                x_km, y_km, math.cos(heading + a), math.sin(heading + a), max_dist_km, bounds_as_wall
            )
            if h is not None:
                out[i] = h.distance_km / max_dist_km
        return out


class MapGenerator:
    """Handles file parsing, coordinate mapping, and procedural noise calculations."""
    def __init__(self, file_path: str | Path, ports_path: str | Path = None):
        self.file_path = Path(file_path)
        self.ports_path = Path(ports_path) if ports_path else None
        if not self.file_path.exists():
            raise FileNotFoundError(f"NetCDF source file not found at: {self.file_path}")

    def generate_map(
        self,
        resolution_km: float,
        bounding_box: dict,
        biomass_seed: int = 42,
        port_ocean_radius_km: float = 4.0,         # Parameterized circular ocean channel radius
        port_ocean_depth_m: float = -50.0,         # Ocean depth set for harbor basin
        port_biomass_clear_radius_km: float = 7.0  # Zero-biomass radius around harbors
    ) -> Map:
        """
        Parses NetCDF grids, projects coordinates to metric kilometers, filters regional ports,
        and executes procedural overrides using circular numpy masks to clear ocean channels and biomass.
        """
        ds = xr.open_dataset(self.file_path)
        height_var = "elevation" if "elevation" in ds else "height"

        ds_subset = ds.sel(
            lon=slice(bounding_box["lon_min"], bounding_box["lon_max"]),
            lat=slice(bounding_box["lat_min"], bounding_box["lat_max"]),
        )

        lon_min, lon_max = float(ds_subset.lon.min()), float(ds_subset.lon.max())
        lat_min, lat_max = float(ds_subset.lat.min()), float(ds_subset.lat.max())

        center_lat = (lat_min + lat_max) / 2.0
        lon_scale = np.cos(np.radians(center_lat))

        width_km = (lon_max - lon_min) * KM_PER_DEGREE * lon_scale
        height_km = (lat_max - lat_min) * KM_PER_DEGREE

        x_km = np.arange(0, width_km, resolution_km)
        y_km = np.arange(0, height_km, resolution_km)
        lons = lon_min + x_km / (KM_PER_DEGREE * lon_scale)
        lats = lat_min + y_km / KM_PER_DEGREE
        lons2d, lats2d = np.meshgrid(lons, lats)

        height_grid = ds_subset[height_var].interp(
            lon=xr.DataArray(lons2d, dims=("y", "x")),
            lat=xr.DataArray(lats2d, dims=("y", "x")),
            method="linear",
        ).values

        biomass_grid = self._generate_biomass_noise(height_grid, seed=biomass_seed)

        # Parse ports list from disk
        filtered_ports = []
        if self.ports_path and self.ports_path.exists():
            with open(self.ports_path, "r", encoding="utf-8") as f:
                ports_list = json.load(f)
            for port in ports_list:
                # SKIP RIVER ENTRIES ENTIRELY
                # if port.get("type") == "river":
                #     continue

                lat, lon = port["lat"], port["lon"]
                if bounding_box["lon_min"] <= lon <= bounding_box["lon_max"] and bounding_box["lat_min"] <= lat <= bounding_box["lat_max"]:
                    filtered_ports.append({
                        "name": port["name"].split("(")[0].strip(),  # strip labels
                        "grid_x": (lon - bounding_box["lon_min"]) * KM_PER_DEGREE * lon_scale,
                        "grid_y": (lat - bounding_box["lat_min"]) * KM_PER_DEGREE,
                        "jurisdiction": port.get("jurisdiction", ""),
                    })

        # =========================================================================
        # PROCEDURAL HARBOUR CHANNEL MODIFICATIONS (Fully Circular Grid Passes)
        # =========================================================================
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))

        # Build kilometer coordinate matrices matching pixel grid indices
        x_coords_km = xi * resolution_km
        y_coords_km = yi * resolution_km

        for port in filtered_ports:
            px_km, py_km = port["grid_x"], port["grid_y"]

            # Calculate continuous Euclidean distances to the center of this specific port
            distance_map_km = np.hypot(x_coords_km - px_km, y_coords_km - py_km)

            # Override A: Dig a circular harbor channel basin into the terrain matrix
            ocean_circle_mask = distance_map_km <= port_ocean_radius_km
            height_grid[ocean_circle_mask] = port_ocean_depth_m

            # Override B: Wipe out biomass hotspot concentrations in a circle
            biomass_circle_mask = distance_map_km <= port_biomass_clear_radius_km
            biomass_grid[biomass_circle_mask] = 0.0

        # Map construction builds the land mask, shore distance, and port flow fields
        # from the carved terrain.
        return Map(height_grid, biomass_grid, width_km, height_km, filtered_ports)

    def _generate_biomass_noise(self, height_grid: np.ndarray, seed: int) -> np.ndarray:
        mask = (height_grid >= -220) & (height_grid <= -80)
        ramp = np.clip(distance_transform_edt(mask) / 5.0, 0.0, 1.0) if np.any(mask) else np.zeros_like(height_grid, dtype=float)
        gen = OpenSimplex(seed=seed)
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        noise = np.vectorize(gen.noise2)(xi / 30.0, yi / 30.0)
        return ramp * ((noise + 1.0) / 2.0)