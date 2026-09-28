import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt  # Required for the SDF edge-fading

import json


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

def get_active_simulation_ports(ports_list: list, bounds: dict):
    """
    Filters master port entries to active map bounds and transforms 
    GPS degrees to exact flat simulator grid [X_km, Y_km] markers.
    """
    km_per_degree = 111.32
    center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2.0
    lon_scale_factor = np.cos(np.radians(center_lat))
    
    filtered_ports = []
    for port in ports_list:
        lat, lon = port['lat'], port['lon']
        
        # Check boundary collision
        if (bounds['lon_min'] <= lon <= bounds['lon_max']) and (bounds['lat_min'] <= lat <= bounds['lat_max']):
            x_km = (lon - bounds['lon_min']) * km_per_degree * lon_scale_factor
            y_km = (lat - bounds['lat_min']) * km_per_degree
            
            filtered_ports.append({
                "name": port['name'],
                "grid_x": x_km,
                "grid_y": y_km,
                "jurisdiction": port['jurisdiction']
            })
    return filtered_ports

def render_marl_map(marl_height_grid: np.ndarray, hotspot_overlay: np.ndarray, width_km: float, height_km: float, grid_res_km: float, bounding_box: dict, active_ports: list):
    plt.figure(figsize=(11, 9), dpi=100)
    v_min, v_max = -5000.0, 2000.0
    total_range = v_max - v_min
    def get_pos(val): return (val - v_min) / total_range
        
    cdict = [
        (get_pos(-5000), "#000814"), (get_pos(-3500), "#001d3d"), (get_pos(-2000), "#003566"),
        (get_pos(-500),  "#1e6091"), (get_pos(0),     "#4cc9f0"), 
        (get_pos(0),     "#a3b18a"), (get_pos(500),   "#588157"), (get_pos(2000),  "#344e41")
    ]
    hybrid_cmap = LinearSegmentedColormap.from_list("high_contrast_hybrid", cdict, N=256)
    
    alpha_red_dict = {
        'red': [(0.0, 1.0, 1.0), (1.0, 0.9, 0.9)], 'green': [(0.0, 0.2, 0.2), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 'alpha': [(0.0, 0.0, 0.0), (1.0, 0.85, 0.85)]
    }
    alpha_red_cmap = LinearSegmentedColormap('alpha_red', alpha_red_dict)
    
    ax = plt.gca()
    img = ax.imshow(marl_height_grid, origin='lower', extent=[0, width_km, 0, height_km], cmap=hybrid_cmap, vmin=v_min, vmax=v_max)
    overlay_img = ax.imshow(hotspot_overlay, origin='lower', extent=[0, width_km, 0, height_km], cmap=alpha_red_cmap, vmin=0.0, vmax=1.0)
    
    # --- ZOOM-DEPENDENT TEXT LAYER FILTER ---
    show_labels = True
    
    for port in active_ports:
        # Plot the clean, distinct red-dot asset indicator
        ax.scatter(port['grid_x'], port['grid_y'], color='#ff007f', edgecolor='white', linewidth=1.0, s=40, zorder=5)
        
        # Only draw text vectors if our map bounds are tightly zoomed in
        if show_labels:
            ax.text(
                port['grid_x'] + 4.0, port['grid_y'] - 2.0, 
                port['name'], 
                color='white', 
                fontsize=7.5,      # Smaller, cleaner font
                fontweight='medium', 
                alpha=0.85, 
                zorder=6,
                bbox=dict(facecolor='black', alpha=0.3, lw=0, pad=1) # Minimal padding background
            )
        
    km_per_degree = 111.32
    center_lat = (bounding_box['lat_min'] + bounding_box['lat_max']) / 2.0
    lon_scale_factor = np.cos(np.radians(center_lat))
    
    def k_to_lon(x): return bounding_box['lon_min'] + (x / (km_per_degree * lon_scale_factor))
    def lon_to_k(lon): return (lon - bounding_box['lon_min']) * km_per_degree * lon_scale_factor
    def k_to_lat(y): return bounding_box['lat_min'] + (y / km_per_degree)
    def lat_to_k(lat): return (lat - bounding_box['lat_min']) * km_per_degree

    secax_x = ax.secondary_xaxis('top', functions=(k_to_lon, lon_to_k))
    secax_x.set_xlabel('Longitude (°W / °E)', fontsize=11, labelpad=10)
    secax_y = ax.secondary_yaxis('right', functions=(k_to_lat, lat_to_k))
    secax_y.set_ylabel('Latitude (°N)', fontsize=11, labelpad=15, rotation=270)

    ax.set_xlabel("X Coordinate (Kilometers from West Bound)", fontsize=11)
    ax.set_ylabel("Y Coordinate (Kilometers from South Bound)", fontsize=11)
    plt.title(f"MARL Fishing Environment Map (Flattened Grid)\nResolution: {grid_res_km}km x {grid_res_km}km per cell", fontsize=14, fontweight='bold', pad=35)
    
    ticks = [-5000, -3500, -2000, -500, 0, 500, 2000]
    cbar = plt.colorbar(img, orientation='vertical', shrink=0.65, pad=0.12, ticks=ticks)
    cbar.set_label('Elevation / Depth (Meters)', rotation=270, labelpad=15)
    
    cbar_hotspot = plt.colorbar(overlay_img, orientation='horizontal', shrink=0.5, pad=0.12)
    cbar_hotspot.set_label('Biomass / Fishing Effort Density (SDF Soft Edge Transition)', fontsize=10)
    
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.15)
    plt.tight_layout()
    plt.show()


def load_and_interpolate_grid(file_path: str, grid_res_km: float = 1.0, bounding_box: dict = None):
    """
    Loads NetCDF bathymetry data, optionally slices it to a coastal subset rectangle,
    flattens coordinates to a KM grid, and interpolates heights.
    
    bounding_box expects: {'lon_min': -125.0, 'lon_max': -118.0, 'lat_min': 35.0, 'lat_max': 45.0}
    """
    ds = xr.open_dataset(file_path)
    height_var = 'elevation' if 'elevation' in ds else 'height'
    
    # --- NEW: Subset the map to a coastal bounding box if provided ---
    if bounding_box is not None:
        # Note: Depending on your NC file, latitude coordinates might be flipped. 
        # slice(min, max) works fine if the dataset coordinates are monotonically increasing.
        ds = ds.sel(
            lon=slice(bounding_box['lon_min'], bounding_box['lon_max']),
            lat=slice(bounding_box['lat_min'], bounding_box['lat_max'])
        )
    
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

# =========================================================================
# RUNTIME EXECUTION
# =========================================================================
if __name__ == "__main__":
    file_path = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
    ports_file_path = "datasets/ports.json"  # <-- Text path definition
    RESOLUTION = 1.0 
    
    coastal_subset = {
        'lon_min': -126.0,  
        'lon_max': -122.0,
        'lat_min': 42.0,
        'lat_max': 50.0     
    }
    
    # Step 1: Compute flat bathymetry grid topology
    height_grid, width_km, height_km = load_and_interpolate_grid(
        file_path, 
        grid_res_km=RESOLUTION, 
        bounding_box=coastal_subset
    )
    
    # Step 2: Generate dynamic hotspots with an SDF edge fade
    hotspot_overlay = generate_dynamic_hotspots(
        marl_height_grid=height_grid, 
        seed=42, 
        noise_feature_scale=30.0, 
        fade_margin_px=5.0
    )
    
    # --- NEW STEP 3: Load the raw JSON text file dataset ---
    with open(ports_file_path, "r", encoding="utf-8") as f:
        fishing_ports_dataset = json.load(f)
        
    # Step 4: Extract and filter ports into the simulation kilometer coordinates
    active_simulation_ports = get_active_simulation_ports(
        ports_list=fishing_ports_dataset, 
        bounds=coastal_subset
    )
    
    # Step 5: Render environment pipeline with port vectors overlaid
    render_marl_map(
        marl_height_grid=height_grid, 
        hotspot_overlay=hotspot_overlay, 
        width_km=width_km, 
        height_km=height_km, 
        grid_res_km=RESOLUTION,
        bounding_box=coastal_subset,
        active_ports=active_simulation_ports
    )
