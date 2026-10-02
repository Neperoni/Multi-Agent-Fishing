# multiagent_fishing/map_generator.py

from pathlib import Path
import json
import numpy as np
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt

KM_PER_DEGREE = 111.32

class Map:
    """
    Represents a localized grid containing underlying height data, generated biological/biomass 
    densities, and structural simulation ports. Handles spatial calculations in kilometers.
    """
    def __init__(self, height_grid: np.ndarray, biomass_grid: np.ndarray, width_km: float, height_km: float, ports: list[dict]):
        self._height_grid = height_grid
        self._biomass_grid = biomass_grid
        self.width_km = width_km
        self.height_km = height_km
        self.rows, self.cols = height_grid.shape
        self._ports = ports

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


class MapGenerator:
    """Handles file parsing, coordinate mapping, and procedural noise calculations."""
    def __init__(self, file_path: str | Path, ports_path: str | Path = None):
        self.file_path = Path(file_path)
        self.ports_path = Path(ports_path) if ports_path else None
        if not self.file_path.exists():
            raise FileNotFoundError(f"NetCDF source file not found at: {self.file_path}")

    def get_map(
        self, 
        resolution_km: float, 
        bounding_box: dict, 
        biomass_seed: int = 42,
        port_ocean_radius_km: float = 2.0,         # Parameterized circular ocean channel radius
        port_ocean_depth_m: float = -50.0,         # Ocean depth set for harbor basin
        port_biomass_clear_radius_km: float = 3.0  # Zero-biomass radius around harbors
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
                lat, lon = port["lat"], port["lon"]
                if bounding_box["lon_min"] <= lon <= bounding_box["lon_max"] and bounding_box["lat_min"] <= lat <= bounding_box["lat_max"]:
                    filtered_ports.append({
                        "name": port["name"].split("(")[0].strip(),
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

            # Override B: Wipe out biomass hotspot concentrations in a 3km circle
            biomass_circle_mask = distance_map_km <= port_biomass_clear_radius_km
            biomass_grid[biomass_circle_mask] = 0.0

        return Map(height_grid, biomass_grid, width_km, height_km, filtered_ports)

    def _generate_biomass_noise(self, height_grid: np.ndarray, seed: int) -> np.ndarray:
        mask = (height_grid >= -220) & (height_grid <= -80)
        ramp = np.clip(distance_transform_edt(mask) / 5.0, 0.0, 1.0) if np.any(mask) else np.zeros_like(height_grid, dtype=float)
        gen = OpenSimplex(seed=seed)
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        noise = np.vectorize(gen.noise2)(xi / 30.0, yi / 30.0)
        return ramp * ((noise + 1.0) / 2.0)
