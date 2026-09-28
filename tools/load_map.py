import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt  # Required for the SDF edge-fading

def load_and_interpolate_grid(file_path: str, grid_res_km: float = 1.0):
    """
    Loads NetCDF bathymetry data, flattens coordinates to a KM grid based on 
    the center latitude, and interpolates heights.
    """
    ds = xr.open_dataset(file_path)
    height_var = 'elevation' if 'elevation' in ds else 'height'
    
    lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
    lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
    
    # Calculate center-scaled flat parameters
    center_lat = (lat_min + lat_max) / 2.0
    km_per_degree = 111.32
    lon_scale_factor = np.cos(np.radians(center_lat))
    
    width_km = (lon_max - lon_min) * km_per_degree * lon_scale_factor
    height_km = (lat_max - lat_min) * km_per_degree
    
    # Build the flat grid coordinates
    x_km = np.arange(0, width_km, grid_res_km)
    y_km = np.arange(0, height_km, grid_res_km)
    
    marl_lons_1d = lon_min + (x_km / (km_per_degree * lon_scale_factor))
    marl_lats_1d = lat_min + (y_km / km_per_degree)
    marl_lons_2d, marl_lats_2d = np.meshgrid(marl_lons_1d, marl_lats_1d)
    
    interp_lons = xr.DataArray(marl_lons_2d, dims=("y", "x"))
    interp_lats = xr.DataArray(marl_lats_2d, dims=("y", "x"))
    
    marl_height_grid = ds[height_var].interp(lon=interp_lons, lat=interp_lats, method="linear").values
    
    return marl_height_grid, width_km, height_km

def generate_dynamic_hotspots(
    marl_height_grid: np.ndarray, 
    seed: int = 42, 
    noise_feature_scale: float = 30.0,
    fade_margin_px: float = 5.0
):
    """
    Generates an upwelling biomass mask based on depth constraints, builds an SDF 
    to smoothly attenuate mask edges to 0, and modulates it with Simplex noise.
    """
    # 1. Baseline binary mask
    upwelling_fall_mask = (marl_height_grid >= -220) & (marl_height_grid <= -80)
    
    # 2. Compute Signed Distance Field (distance transform) to smoothly fade edges
    # distance_transform_edt computes the distance to the closest 0 (outside mask)
    if np.any(upwelling_fall_mask):
        sdf_grid = distance_transform_edt(upwelling_fall_mask)
        # Create a smooth normalized ramp [0.0 to 1.0] over the defined margin size
        smooth_edge_ramp = np.clip(sdf_grid / fade_margin_px, 0.0, 1.0)
    else:
        smooth_edge_ramp = np.zeros_like(marl_height_grid, dtype=float)

    # 3. Generate Simplex Noise mapping to [0.0, 1.0]
    gen = OpenSimplex(seed=seed)
    ny, nx = marl_height_grid.shape
    y_indices, x_indices = np.indices((ny, nx))
    
    noise_grid = np.vectorize(gen.noise2)(x_indices / noise_feature_scale, y_indices / noise_feature_scale)
    noise_grid = (noise_grid + 1.0) / 2.0
    
    # 4. Multiply mask-bound, edge-attenuated SDF by noise values
    dynamic_hotspots = smooth_edge_ramp * noise_grid
    
    return dynamic_hotspots

def render_marl_map(marl_height_grid: np.ndarray, hotspot_overlay: np.ndarray, width_km: float, height_km: float, grid_res_km: float):
    """
    Plots the environmental bathymetry with the custom dynamic alpha-faded hotspot overlay.
    """
    plt.figure(figsize=(10, 8), dpi=100)
    v_min, v_max = -5000.0, 2000.0
    total_range = v_max - v_min
    
    def get_pos(val): 
        return (val - v_min) / total_range
        
    # Standard environmental depth map custom layout
    cdict = [
        (get_pos(-5000), "#000814"), (get_pos(-3500), "#001d3d"), (get_pos(-2000), "#003566"),
        (get_pos(-220), "#00509d"), (get_pos(-220), "#1e6091"), (get_pos(-80), "#1e6091"),
        (get_pos(-80), "#4cc9f0"), (get_pos(0), "#4cc9f0"), (get_pos(0), "#a3b18a"),
        (get_pos(500), "#a3b18a"), (get_pos(500), "#588157"), (get_pos(2000), "#344e41")
    ]
    hybrid_cmap = LinearSegmentedColormap.from_list("high_contrast_hybrid", cdict, N=256)
    
    # Dynamic alpha custom crimson map for masking
    alpha_red_dict = {
        'red': [(0.0, 1.0, 1.0), (1.0, 0.9, 0.9)],
        'green': [(0.0, 0.2, 0.2), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'alpha': [(0.0, 0.0, 0.0), (1.0, 0.85, 0.85)]
    }
    alpha_red_cmap = LinearSegmentedColormap('alpha_red', alpha_red_dict)
    
    # Plot base bathymetry map
    img = plt.imshow(marl_height_grid, origin='lower', extent=[0, width_km, 0, height_km], cmap=hybrid_cmap, vmin=v_min, vmax=v_max)
    
    # Overlay hotspots using the new dynamic-alpha colormap
    overlay_img = plt.imshow(hotspot_overlay, origin='lower', extent=[0, width_km, 0, height_km], cmap=alpha_red_cmap, vmin=0.0, vmax=1.0)
    
    # Add UI / Aesthetic Elements
    plt.title(f"MARL Fishing Environment Map (Flattened Grid)\nResolution: {grid_res_km}km x {grid_res_km}km per cell", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("X Coordinate (Kilometers from West Bound)", fontsize=11)
    plt.ylabel("Y Coordinate (Kilometers from South Bound)", fontsize=11)
    
    # Colorbars
    ticks = [-5000, -3500, -2000, -220, -80, 0, 500, 2000]
    cbar = plt.colorbar(img, orientation='vertical', shrink=0.75, pad=0.04, ticks=ticks)
    cbar.set_label('Elevation / Depth (Meters)', rotation=270, labelpad=15)
    
    cbar_hotspot = plt.colorbar(overlay_img, orientation='horizontal', shrink=0.5, pad=0.08)
    cbar_hotspot.set_label('Biomass / Fishing Effort Density (SDF Soft Edge Transition)', fontsize=10)
    
    plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.tight_layout()
    plt.show()

# =========================================================================
# RUNTIME EXECUTION
# =========================================================================
if __name__ == "__main__":
    file_path = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
    RESOLUTION = 2.0  # 1km grid res
    
    # Step 1: Compute flat bathymetry grid
    height_grid, width_km, height_km = load_and_interpolate_grid(file_path, grid_res_km=RESOLUTION)
    
    # Step 2: Generate dynamic hotspots with a 5-pixel SDF edge fade
    hotspot_overlay = generate_dynamic_hotspots(
        marl_height_grid=height_grid, 
        seed=42, 
        noise_feature_scale=30.0, 
        fade_margin_px=3.0  # Adjust this to change how soft the edges look
    )
    
    # Step 3: Render pipeline
    render_marl_map(height_grid, hotspot_overlay, width_km, height_km, grid_res_km=RESOLUTION)
