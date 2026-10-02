# multiagent_fishing/map_generator.py

from pathlib import Path
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

    # =========================================================================
    # COORDINATE SPACE BOUNDARY CODES & SAMPLING METHODS
    # =========================================================================
    def is_in_bounds(self, x_km: float, y_km: float) -> bool:
        """Checks if a continuous kilometer coordinate point rests inside the active tracking canvas."""
        return 0.0 <= x_km <= self.width_km and 0.0 <= y_km <= self.height_km

    def get_height_at_pixel(self, x_km: float, y_km: float, method: str = "closest") -> float:
        """Gets the elevation/depth data directly from a continuous coordinate coordinate string."""
        return self._sample_grid(self._height_grid, x_km, y_km, method)

    def get_biomass_at_pixel(self, x_km: float, y_km: float, method: str = "closest") -> float:
        """Gets the biomass/hotspot density directly from a continuous coordinate coordinate string."""
        return self._sample_grid(self._biomass_grid, x_km, y_km, method)

    def km_to_pixel_coords(self, x_km: float, y_km: float) -> tuple[float, float]:
        """Converts continuous kilometer coordinates into raw continuous floating-point cell indices."""
        res = self.km_per_pixel
        if res <= 0.0:
            return 0.0, 0.0
        return x_km / res, y_km / res

    def _sample_grid(self, grid: np.ndarray, x_km: float, y_km: float, method: str) -> float:
        """Internal worker method to handle closest cell indexing or bilinear interpolation matrices."""
        if not self.is_in_bounds(x_km, y_km):
            raise IndexError(f"Kilometer coordinates ({x_km:.2f}, {y_km:.2f}) out of bounds for map dimensions ({self.width_km:.1f}km, {self.height_km:.1f}km)")

        # Convert kilometer vector points into continuous pixel spaces
        px, py = self.km_to_pixel_coords(x_km, y_km)

        if method == "closest":
            col = min(int(np.floor(px)), self.cols - 1)
            row = min(int(np.floor(py)), self.rows - 1)
            return float(grid[row, col])

        elif method == "bilinear":
            # 1. Grab bounding floor indices
            c0 = int(np.floor(px))
            r0 = int(np.floor(py))
            
            # 2. Grab ceiling limits while protecting matrix array edges
            c1 = min(c0 + 1, self.cols - 1)
            r1 = min(r0 + 1, self.rows - 1)

            # 3. Calculate interpolation ratios (weights)
            tx = px - c0
            ty = py - r0

            # 4. Perform standard two-axis bilinear interpolation blending math
            val_bottom = (1.0 - tx) * grid[r0, c0] + tx * grid[r0, c1]
            val_top = (1.0 - tx) * grid[r1, c0] + tx * grid[r1, c1]
            
            return float((1.0 - ty) * val_bottom + ty * val_top)

        else:
            raise ValueError(f"Unknown interpolation type format requested: '{method}'. Use 'closest' or 'bilinear'.")


class MapGenerator:
    """Handles file parsing, coordinate mapping, and procedural noise calculations."""
    def __init__(self, file_path: str | Path, ports_path: str | Path = None):
        self.file_path = Path(file_path)
        self.ports_path = Path(ports_path) if ports_path else None
        if not self.file_path.exists():
            raise FileNotFoundError(f"NetCDF source file not found at: {self.file_path}")

    def get_map(self, resolution_km: float, bounding_box: dict, biomass_seed: int = 42) -> Map:
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

        filtered_ports = []
        if self.ports_path and self.ports_path.exists():
            import json
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

        return Map(height_grid, biomass_grid, width_km, height_km, filtered_ports)

    def _generate_biomass_noise(self, height_grid: np.ndarray, seed: int) -> np.ndarray:
        mask = (height_grid >= -220) & (height_grid <= -80)
        ramp = np.clip(distance_transform_edt(mask) / 5.0, 0.0, 1.0) if np.any(mask) else np.zeros_like(height_grid, dtype=float)
        gen = OpenSimplex(seed=seed)
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        noise = np.vectorize(gen.noise2)(xi / 30.0, yi / 30.0)
        return ramp * ((noise + 1.0) / 2.0)
