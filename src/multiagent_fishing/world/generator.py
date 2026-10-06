import json
import math
from pathlib import Path

import numpy as np
import skfmm
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt


from .map import KM_PER_DEGREE, SEA_LEVEL, Layers, Map

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
        port_biomass_clear_radius_km: float = 7.0,  # Zero-biomass radius around harbors
        shore_penalty_km: float = 1.0,
        shore_penalty_strength: float = 2.0
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
        port_dist, port_flow = self._port_fields(
            land_mask, shore_dist, ports, resolution_km, cap_km,
            shore_penalty_km, shore_penalty_strength,
        )
        nearest_id, nearest_dist, nearest_dir = self._nearest_port_layers(port_dist, port_flow, cap_km)

        by_channel = {
            Layers.HEIGHT: height,
            Layers.BIOMASS: biomass,
            Layers.SHORE_DIST: shore_dist,
            Layers.NEAREST_PORT_ID: nearest_id,
            Layers.NEAREST_PORT_DIST: nearest_dist,
            Layers.NEAREST_PORT_DIR_X: nearest_dir[0],
            Layers.NEAREST_PORT_DIR_Y: nearest_dir[1],
        }
        layers = np.stack([by_channel[c] for c in Layers], axis=-1).astype(np.float32)

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
    
    def _port_fields(
        self,
        land_mask: np.ndarray,
        shore_dist: np.ndarray,
        ports: list[dict],
        resolution_km: float,
        cap_km: float,
        shore_penalty_km: float,
        shore_penalty_strength: float,
    ):
        """
        Two Fast Marching solves per port over water cells (any-angle):
          - unweighted -> port_dist, true path length in km
          - shore-averse weighted -> port_flow only
        Returns port_dist (n, rows, cols) float32 in km, capped at cap_km where unreachable,
        and port_flow (n, 2, rows, cols) float16 unit vectors toward the port.
        """
        rows, cols = land_mask.shape
        dist = np.full((len(ports), rows, cols), cap_km, dtype=np.float32)
        flow = np.zeros((len(ports), 2, rows, cols), dtype=np.float16)

        # Cost is 1 in open water and rises to 1 + strength next to land,
        # fading back to 1 after `shore_penalty_km` further out.
        width = max(shore_penalty_km, resolution_km)
        falloff = np.clip(1.0 - (shore_dist - resolution_km) / width, 0.0, 1.0)
        speed_base = 1.0 / (1.0 + shore_penalty_strength * falloff)

        for i, port in enumerate(ports):
            col = min(max(int(port["grid_x"] / resolution_km), 0), cols - 1)
            row = min(max(int(port["grid_y"] / resolution_km), 0), rows - 1)

            blocked = land_mask.copy()
            blocked[row, col] = False  # guarantee the start cell is traversable

            def make_phi():
                phi = np.ma.MaskedArray(np.ones((rows, cols)), mask=blocked)
                phi[row, col] = 0.0
                return phi

            # Pure geometric distance: what gets stored as port_dist.
            true_km = np.ma.filled(skfmm.distance(make_phi(), dx=resolution_km), np.inf)

            # Weighted travel time: only used to shape the flow direction.
            speed = np.ma.MaskedArray(speed_base, mask=blocked)
            weighted = np.ma.filled(
                skfmm.travel_time(make_phi(), speed, dx=resolution_km), np.inf
            )

            reachable = np.isfinite(true_km)
            dist[i] = np.where(reachable, np.minimum(true_km, cap_km), cap_km)
            flow[i] = self._flow_from_distance(weighted, reachable, resolution_km).astype(np.float16)

        return dist, flow

    @staticmethod
    def _flow_from_distance(cum: np.ndarray, reachable: np.ndarray, resolution_km: float) -> np.ndarray:
        """
        Unit vector toward the port at any angle: the negative gradient of the distance field.
        Shape (2, rows, cols), (x, y) order, zero where unreachable.
        """
        if not reachable.any():
            return np.zeros((2, *cum.shape))

        # Replace land/unreachable cells with the value of the nearest reachable cell so the
        # gradient near coastlines is not distorted by infinities or artificial cliffs.
        idx = distance_transform_edt(~reachable, return_distances=False, return_indices=True)
        filled = cum[tuple(idx)]

        gy, gx = np.gradient(filled, resolution_km)  # axis 0 = y (rows), axis 1 = x (cols)
        fx, fy = -gx, -gy

        norm = np.hypot(fx, fy)
        ok = reachable & (norm > 1e-9)
        fx = np.where(ok, fx / np.where(ok, norm, 1.0), 0.0)
        fy = np.where(ok, fy / np.where(ok, norm, 1.0), 0.0)
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