import json
import math
from pathlib import Path

import numpy as np
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

from .map import KM_PER_DEGREE, SEA_LEVEL, Channel, Map

# Fish live in this depth band (meters, negative = below sea level)
BIOMASS_DEPTH_MIN = -220.0
BIOMASS_DEPTH_MAX = -80.0

# Cells with no source data (NaN) are treated as shallow water
NAN_HEIGHT_FILL = -1.0


class MapGenerator:
    """Builds a finished Map from a NetCDF height file and an optional ports JSON."""

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
        port_ocean_radius_km: float = 4.0,         # Circular ocean channel radius at each port
        port_ocean_depth_m: float = -50.0,         # Depth of the harbor basin
        port_biomass_clear_radius_km: float = 7.0  # Zero-biomass radius around harbors
    ) -> Map:
        height, width_km, height_km, lon_min, lat_min, lon_scale = self._load_heightmap(
            resolution_km, bounding_box
        )
        height = np.where(np.isnan(height), NAN_HEIGHT_FILL, height)

        biomass = self._generate_biomass(height, biomass_seed)
        ports = self._load_ports(bounding_box, lon_min, lat_min, lon_scale)
        self._carve_harbors(
            height, biomass, ports, resolution_km,
            port_ocean_radius_km, port_ocean_depth_m, port_biomass_clear_radius_km,
        )

        # Terrain edits are done, so everything derived can be computed now.
        land_mask = height >= SEA_LEVEL
        cap_km = math.hypot(width_km, height_km)  # stands in for "no path" / "no land"

        shore_dist = self._shore_distance(land_mask, resolution_km, cap_km)
        port_dist, port_flow = self._port_fields(land_mask, ports, resolution_km, cap_km)
        nearest_id, nearest_dist, nearest_dir = self._nearest_port_layers(port_dist, port_flow, cap_km)

        by_channel = {
            Channel.HEIGHT: height,
            Channel.BIOMASS: biomass,
            Channel.SHORE_DIST: shore_dist,
            Channel.NEAREST_PORT_ID: nearest_id,
            Channel.NEAREST_PORT_DIST: nearest_dist,
            Channel.NEAREST_PORT_DIR_X: nearest_dir[0],
            Channel.NEAREST_PORT_DIR_Y: nearest_dir[1],
        }
        layers = np.stack([by_channel[c] for c in Channel], axis=-1).astype(np.float32)

        return Map(
            layers=layers,
            port_dist=port_dist,
            port_flow=port_flow,
            width_km=width_km,
            height_km=height_km,
            resolution_km=resolution_km,
            ports=ports,
            origin_lon=lon_min,
            origin_lat=lat_min,
            lon_scale=lon_scale,
        )

    # ------------------------------------------------------------------
    # Height data
    # ------------------------------------------------------------------
    def _load_heightmap(self, resolution_km: float, bounding_box: dict):
        """
        Reads the NetCDF, crops to the bounding box, and resamples onto a regular km grid.
        Returns (height_grid, width_km, height_km, lon_min, lat_min, lon_scale).
        """
        with xr.open_dataset(self.file_path) as ds:
            height_var = "elevation" if "elevation" in ds else "height"

            ds_subset = ds.sel(
                lon=slice(bounding_box["lon_min"], bounding_box["lon_max"]),
                lat=slice(bounding_box["lat_min"], bounding_box["lat_max"]),
            )

            lon_min, lon_max = float(ds_subset.lon.min()), float(ds_subset.lon.max())
            lat_min, lat_max = float(ds_subset.lat.min()), float(ds_subset.lat.max())

            center_lat = (lat_min + lat_max) / 2.0
            lon_scale = float(np.cos(np.radians(center_lat)))

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

        return height_grid, width_km, height_km, lon_min, lat_min, lon_scale

    # ------------------------------------------------------------------
    # Biomass
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_biomass(height_grid: np.ndarray, seed: int) -> np.ndarray:
        """Simplex-noise density in [0, 1], limited to the fishable depth band with soft edges."""
        mask = (height_grid >= BIOMASS_DEPTH_MIN) & (height_grid <= BIOMASS_DEPTH_MAX)
        if np.any(mask):
            ramp = np.clip(distance_transform_edt(mask) / 5.0, 0.0, 1.0)
        else:
            ramp = np.zeros_like(height_grid, dtype=float)

        gen = OpenSimplex(seed=seed)
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        noise = np.vectorize(gen.noise2)(xi / 30.0, yi / 30.0)
        return ramp * ((noise + 1.0) / 2.0)

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------
    def _load_ports(self, bounding_box: dict, lon_min: float, lat_min: float, lon_scale: float) -> list[dict]:
        """Reads ports, keeps those inside the bounding box, projects them to map km. Id = index."""
        if not (self.ports_path and self.ports_path.exists()):
            return []

        with open(self.ports_path, "r", encoding="utf-8") as f:
            ports_list = json.load(f)

        ports = []
        for port in ports_list:
            # To skip river entries entirely:
            # if port.get("type") == "river":
            #     continue
            lat, lon = port["lat"], port["lon"]
            if (bounding_box["lon_min"] <= lon <= bounding_box["lon_max"]
                    and bounding_box["lat_min"] <= lat <= bounding_box["lat_max"]):
                ports.append({
                    "id": len(ports),
                    "name": port["name"].split("(")[0].strip(),  # strip labels
                    "grid_x": (lon - lon_min) * KM_PER_DEGREE * lon_scale,
                    "grid_y": (lat - lat_min) * KM_PER_DEGREE,
                    "jurisdiction": port.get("jurisdiction", ""),
                })
        return ports

    @staticmethod
    def _carve_harbors(
        height_grid: np.ndarray,
        biomass_grid: np.ndarray,
        ports: list[dict],
        resolution_km: float,
        ocean_radius_km: float,
        ocean_depth_m: float,
        biomass_clear_radius_km: float,
    ) -> None:
        """In place: dig a circular basin at each port and zero the biomass around it."""
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        x_km = xi * resolution_km
        y_km = yi * resolution_km

        for port in ports:
            dist_km = np.hypot(x_km - port["grid_x"], y_km - port["grid_y"])
            height_grid[dist_km <= ocean_radius_km] = ocean_depth_m
            biomass_grid[dist_km <= biomass_clear_radius_km] = 0.0

    # ------------------------------------------------------------------
    # Derived navigation fields
    # ------------------------------------------------------------------
    @staticmethod
    def _shore_distance(land_mask: np.ndarray, resolution_km: float, cap_km: float) -> np.ndarray:
        """Distance (km) from each cell to the nearest land cell. 0 on land."""
        if not land_mask.any():
            return np.full(land_mask.shape, cap_km)
        return np.minimum(distance_transform_edt(~land_mask) * resolution_km, cap_km)

    def _port_fields(self, land_mask: np.ndarray, ports: list[dict], resolution_km: float, cap_km: float):
        """
        One Dijkstra per port over water cells.
        Returns port_dist (n, rows, cols) float32 in km, capped at cap_km where unreachable,
        and port_flow (n, 2, rows, cols) float16 unit vectors toward the port (0 where unreachable).
        """
        rows, cols = land_mask.shape
        dist = np.full((len(ports), rows, cols), cap_km, dtype=np.float32)
        flow = np.zeros((len(ports), 2, rows, cols), dtype=np.float16)

        for i, port in enumerate(ports):
            col = min(max(int(port["grid_x"] / resolution_km), 0), cols - 1)
            row = min(max(int(port["grid_y"] / resolution_km), 0), rows - 1)

            cost = np.where(land_mask, np.inf, 1.0)
            cost[row, col] = 1.0  # guarantee the start cell is traversable
            cum, _ = MCP_Geometric(cost, fully_connected=True).find_costs([(row, col)])
            cum = cum * resolution_km

            reachable = np.isfinite(cum)
            dist[i] = np.where(reachable, np.minimum(cum, cap_km), cap_km)
            flow[i] = (self._steepest_descent(cum) * reachable).astype(np.float16)

        return dist, flow

    @staticmethod
    def _steepest_descent(cum: np.ndarray) -> np.ndarray:
        """Unit vector toward each cell's lowest-cost neighbor. Shape (2, rows, cols)."""
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

    @staticmethod
    def _nearest_port_layers(port_dist: np.ndarray, port_flow: np.ndarray, cap_km: float):
        """
        Per-cell summary over all ports: id of the nearest by water path (-1 if none reachable),
        its distance, and the flow direction toward it. Returns (id, dist, dir) with dir (2, rows, cols).
        """
        n, rows, cols = port_dist.shape
        if n == 0:
            return (
                np.full((rows, cols), -1.0, dtype=np.float32),
                np.full((rows, cols), cap_km, dtype=np.float32),
                np.zeros((2, rows, cols), dtype=np.float32),
            )

        best = port_dist.argmin(axis=0)
        dist = port_dist.min(axis=0)
        reachable = dist < cap_km
        ids = np.where(reachable, best, -1).astype(np.float32)

        dirs = np.zeros((2, rows, cols), dtype=np.float32)
        for i in range(n):
            m = reachable & (best == i)
            dirs[:, m] = port_flow[i][:, m]
        return ids, dist, dirs