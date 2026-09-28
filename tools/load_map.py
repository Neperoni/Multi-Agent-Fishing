import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mpath
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
    Cleans names to strip out structural parentheses additions.
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
            
            # --- FIXED: Extract index 0 from the split list before stripping ---
            clean_name = port['name'].split('(')[0].strip()
            
            filtered_ports.append({
                "name": clean_name,
                "grid_x": x_km,
                "grid_y": y_km,
                "jurisdiction": port['jurisdiction']
            })
    return filtered_ports

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


def get_rotated_boat_marker(heading_rad: float):
    """
    Creates a custom boat-shaped polygon (square with a pointed bow) 
    and rotates it using a real radian heading.
    0.0 radians points straight UP (North), increasing clockwise.
    """
    # Base vertices for a vessel hull silhouette pointing straight UP (+Y axis)
    vertices = np.array([
        [0.0,  0.6],   # 1. Tip of the bow (front point)
        [0.3,  0.2],   # 2. Right front shoulder
        [0.3, -0.5],   # 3. Right rear stern corner
        [-0.3, -0.5],  # 4. Left rear stern corner
        [-0.3,  0.2],  # 5. Left front shoulder
        [0.0,  0.6]    # 6. Close poly at the bow point
    ])
    
    # Core vector graphics drawing instructions
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY,
    ]
    
    # 2D clockwise rotation transformation matrix
    # Inverted to handle Matplotlib's counter-clockwise native orientation
    c, s = np.cos(-heading_rad), np.sin(-heading_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    # Project our layout array points across the rotation matrix transformation
    rotated_vertices = vertices @ rotation_matrix.T
    
    return mpath.Path(rotated_vertices, codes)
def render_marl_map(
    marl_height_grid: np.ndarray, 
    hotspot_overlay: np.ndarray, 
    width_km: float, 
    height_km: float, 
    grid_res_km: float, 
    bounding_box: dict, 
    active_ports: list,
    fishing_boats: list = None,
    patrol_boats: list = None,
    boat_size: float = 65.0
):
    plt.figure(figsize=(12, 9), dpi=100)
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
    
    # --- RENDER PORTS & TRACK TEXT OBJECTS ---
    dynamic_text_objects = []  # Consolidated array tracking all text nodes
    
    for port in active_ports:
        ax.scatter(port['grid_x'], port['grid_y'], color='#ff007f', edgecolor='white', linewidth=1.0, s=40, zorder=5)
        t_obj = ax.text(
            port['grid_x'] + 4.0, port['grid_y'] - 2.0, port['name'], 
            color='white', fontsize=7.5, fontweight='bold', alpha=0.85, zorder=6,
            bbox=dict(facecolor='black', alpha=0.3, lw=0, pad=1)
        )
        dynamic_text_objects.append(t_obj)

    # Vector Path Shape Configuration Builder
    def get_rotated_boat_marker(heading_rad: float):
        vertices = np.array([[0.0, 0.6], [0.3, 0.2], [0.3, -0.5], [-0.3, -0.5], [-0.3, 0.2], [0.0, 0.6]])
        codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
        c, s = np.cos(-heading_rad), np.sin(-heading_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return mpath.Path(vertices @ rotation_matrix.T, codes)

    # --- RENDER FISHING FLEET & TRACK EXTRACTED NAMES ---
    if fishing_boats is not None:
        for idx, boat in enumerate(fishing_boats):
            heading_rad = boat.get('heading', 0.0)
            ax.scatter(
                boat['x_km'], boat['y_km'], color='#ffcc00', edgecolor='black', 
                marker=get_rotated_boat_marker(heading_rad), s=boat_size, linewidth=0.8, zorder=10, 
                label='Fishing Boat' if idx == 0 else ""
            )
            
            boat_name = boat.get('name', f"F{idx + 1}")
            fb_text = ax.text(
                boat['x_km'] + 3.5, boat['y_km'] + 2.5, boat_name, 
                color='#ffcc00', fontsize=7, fontweight='bold', alpha=0.9, zorder=12,
                bbox=dict(facecolor='#050505', alpha=0.5, lw=0, pad=0.8)
            )
            dynamic_text_objects.append(fb_text) # Appended to dynamic handling array

    # --- RENDER PROTECTION PATROL UNITS & TRACK EXTRACTED NAMES ---
    if patrol_boats is not None:
        for idx, patrol in enumerate(patrol_boats):
            heading_rad = patrol.get('heading', 0.0)
            ax.scatter(
                patrol['x_km'], patrol['y_km'], color='#00ff66', edgecolor='black', 
                marker=get_rotated_boat_marker(heading_rad), s=boat_size * 1.15, linewidth=1.0, zorder=11, 
                label='Patrol Boat' if idx == 0 else ""
            )
            
            patrol_name = patrol.get('name', f"P{idx + 1}")
            pb_text = ax.text(
                patrol['x_km'] + 3.5, patrol['y_km'] + 2.5, patrol_name, 
                color='#00ff66', fontsize=7, fontweight='bold', alpha=0.9, zorder=12,
                bbox=dict(facecolor='#050505', alpha=0.5, lw=0, pad=0.8)
            )
            dynamic_text_objects.append(pb_text) # Appended to dynamic handling array
            
    if fishing_boats or patrol_boats:
        ax.legend(
            loc='upper left', bbox_to_anchor=(1.25, 1.0), borderaxespad=0.,
            framealpha=0.8, facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=9.5
        )

    # --- UPDATED INTERACTIVE VIEWPORT SCALING REGISTRY ---
    def on_zoom_change(axes):
        xlim = axes.get_xlim()
        current_visible_width_km = abs(xlim[1] - xlim[0]) # Double indexed fixed array bounds math
        DYNAMIC_THRESHOLD_KM = 400.0
        labels_visible = (current_visible_width_km <= DYNAMIC_THRESHOLD_KM)
        
        # Toggles ports, fishing boat labels, and patrol boat labels simultaneously
        for text_item in dynamic_text_objects:
            text_item.set_visible(labels_visible)
        axes.figure.canvas.draw_idle()

    ax.callbacks.connect('xlim_changed', on_zoom_change)
    on_zoom_change(ax)
        
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
    cbar = plt.colorbar(img, orientation='vertical', shrink=0.65, pad=0.15, ticks=ticks)
    cbar.set_label('Elevation / Depth (Meters)', rotation=270, labelpad=15)
    
    cbar_hotspot = plt.colorbar(overlay_img, orientation='horizontal', shrink=0.5, pad=0.12)
    cbar_hotspot.set_label('Biomass / Fishing Effort Density (SDF Soft Edge Transition)', fontsize=10)
    
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.15)
    plt.tight_layout()
    plt.show()

# =========================================================================
# RUNTIME EXECUTION
# =========================================================================
if __name__ == "__main__":
    file_path = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
    ports_file_path = "datasets/ports.json"
    RESOLUTION = 1.0 
    
    coastal_subset = {
        'lon_min': -126.0, 'lon_max': -122.0,
        'lat_min': 42.0,   'lat_max': 50.0     
    }
    
    height_grid, width_km, height_km = load_and_interpolate_grid(file_path, grid_res_km=RESOLUTION, bounding_box=coastal_subset)
    hotspot_overlay = generate_dynamic_hotspots(marl_height_grid=height_grid, seed=42, noise_feature_scale=30.0, fade_margin_px=5.0)
    
    with open(ports_file_path, "r", encoding="utf-8") as f:
        fishing_ports_dataset = json.load(f)
    active_simulation_ports = get_active_simulation_ports(fishing_ports_dataset, coastal_subset)
    
    sample_fishing_boats = [
        {"x_km": 45.0,  "y_km": 120.0, "heading": np.pi / 4,  "name": "ORION"},
        {"x_km": 90.0,  "y_km": 340.0, "heading": np.pi,      "name": "MARINER"},
        {"x_km": 150.0, "y_km": 510.0, "heading": 1.5 * np.pi, "name": "SEAWOLF"}
    ]
    
    sample_patrol_boats = [
        {"x_km": 60.0,  "y_km": 150.0, "heading": 0.0,         "name": "PATROL_A"},
        {"x_km": 110.0, "y_km": 400.0, "heading": 0.75 * np.pi, "name": "PATROL_B"}
    ]

    # Render map pipeline including vehicles
    render_marl_map(
        marl_height_grid=height_grid, 
        hotspot_overlay=hotspot_overlay, 
        width_km=width_km, 
        height_km=height_km, 
        grid_res_km=RESOLUTION,
        bounding_box=coastal_subset,
        active_ports=active_simulation_ports,
        fishing_boats=sample_fishing_boats, # Passed here
        patrol_boats=sample_patrol_boats    # Passed here
    )
