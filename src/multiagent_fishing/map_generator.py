from pathlib import Path
import numpy as np
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt
import json

KM_PER_DEGREE = 111.32

class Map:
    """
    Represents a localized grid containing underlying height data, 
    generated biological/biomass densities, and structural simulation ports.
    """
    def __init__(self, height_grid: np.ndarray, biomass_grid: np.ndarray, width_km: float, height_km: float, ports: list[dict]):
        self._height_grid = height_grid
        self._biomass_grid = biomass_grid
        self.width_km = width_km
        self.height_km = height_km
        self.rows, self.cols = height_grid.shape
        self._ports = ports  # Store the processed simulation ports here

    @property
    def ports(self) -> list[dict]:
        """Exposes the filtered list of ports inside this map's bounding box region."""
        return self._ports

    def get_height_at_pixel(self, x: int, y: int) -> float:
        """Get the elevation/depth value at specific array grid coordinates."""
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return float(self._height_grid[y, x])
        raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds for map size ({self.cols}, {self.rows})")

    def get_biomass_at_pixel(self, x: int, y: int) -> float:
        """Get the biomass/hotspot value at specific array grid coordinates."""
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return float(self._biomass_grid[y, x])
        raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds for map size ({self.cols}, {self.rows})")

    @property
    def shape(self):
        """Returns the matrix dimensions (rows, cols)."""
        return self._height_grid.shape

    @property
    def km_per_pixel(self) -> float:
        """
        Gets the resolution scale of the map grid cells.
        Returns the horizontal step size in kilometers per pixel.
        """
        if self.cols > 1:
            return self.width_km / self.cols
        return 0.0

    @property
    def height_data(self) -> np.ndarray:
        """Exposes the underlying height grid matrix."""
        return self._height_grid

    @property
    def biomass_data(self) -> np.ndarray:
        """Exposes the underlying biomass grid matrix."""
        return self._biomass_grid

class MapGenerator:
    """
    Handles file parsing, coordinate mapping, structural interpolations, 
    and procedural noise calculations to generate decoupled Map data instances.
    """
    def __init__(self, file_path: str | Path, ports_path: str | Path = None):
        self.file_path = Path(file_path)
        self.ports_path = Path(ports_path) if ports_path else None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"NetCDF source file not found at: {self.file_path}")

    def get_map(self, resolution_km: float, bounding_box: dict, biomass_seed: int = 42) -> Map:
        """
        Parses regional subsets, interpolates grid geometry uniformly to kilometers,
        filters active ports, and generates procedural biomass distributions.
        """
        # --- (Keep your existing data loading, lon_scale math, and height_grid interpolation code) ---
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
        # --- (End of existing code block) ---

        # 4. Process and filter geographic ports within the regional subset box
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

        # 5. Pack data into the Map container instance
        return Map(height_grid, biomass_grid, width_km, height_km, filtered_ports)

    def _generate_biomass_noise(
        self, 
        height_grid: np.ndarray, 
        seed: int, 
        noise_feature_scale: float = 30.0, 
        fade_margin_px: float = 5.0
    ) -> np.ndarray:
        """Calculates structured marine shelf masks merged over deterministic Simplex noise."""
        mask = (height_grid >= -220) & (height_grid <= -80)
        
        if np.any(mask):
            ramp = np.clip(distance_transform_edt(mask) / fade_margin_px, 0.0, 1.0)
        else:
            ramp = np.zeros_like(height_grid, dtype=float)

        gen = OpenSimplex(seed=seed)
        ny, nx = height_grid.shape
        yi, xi = np.indices((ny, nx))
        
        # Vectorized noise pass mapping array grid indices onto the coordinate generator
        noise_func = np.vectorize(gen.noise2)
        noise = noise_func(xi / noise_feature_scale, yi / noise_feature_scale)
        noise = (noise + 1.0) / 2.0
        
        return ramp * noise
