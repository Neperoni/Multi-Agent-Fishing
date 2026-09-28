import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. Load dataset & extract extents
file_path = r"C:\Users\pooti\OneDrive\Desktop\Projects\Multi-Agent-Fishing\datasets\gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
ds = xr.open_dataset(file_path)
height_var = 'elevation' if 'elevation' in ds else 'height'

lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())

# 2. Calculate center-scaled flat parameters
center_lat = (lat_min + lat_max) / 2.0
km_per_degree = 111.32 
lon_scale_factor = np.cos(np.radians(center_lat)) 

width_km = (lon_max - lon_min) * km_per_degree * lon_scale_factor
height_km = (lat_max - lat_min) * km_per_degree

# 3. Build the flat MARL Grid (1km x 1km cells for high resolution)
grid_res_km = 1.0
x_km = np.arange(0, width_km, grid_res_km)
y_km = np.arange(0, height_km, grid_res_km)

marl_lons_1d = lon_min + (x_km / (km_per_degree * lon_scale_factor))
marl_lats_1d = lat_min + (y_km / km_per_degree)

# Force xarray to evaluate this as a full 2D grid matrix intersection
marl_lons_2d, marl_lats_2d = np.meshgrid(marl_lons_1d, marl_lats_1d)
interp_lons = xr.DataArray(marl_lons_2d, dims=("y", "x"))
interp_lats = xr.DataArray(marl_lats_2d, dims=("y", "x"))

# Interp heights to our clean high-res matrix
marl_height_grid = ds[height_var].interp(lon=interp_lons, lat=interp_lats, method="linear").values

# 4. Generate hot spot mask (Updated to match your adjusted parameters)
upwelling_fall_mask = (marl_height_grid >= -220) & (marl_height_grid <= -80)

# Convert to an overlay grid (replace False with np.nan so it is perfectly transparent)
hotspot_overlay = np.where(upwelling_fall_mask, 1.0, np.nan)

# =========================================================================
# 5. RENDER THE ENVIRONMENT WITH A HYBRID SMOOTH-DEEP / DISCRETE-SHORE MAP
# =========================================================================
plt.figure(figsize=(10, 8), dpi=100)

# Establish anchoring checkpoints for the non-linear map scaling (-5000m to +2000m)
v_min, v_max = -5000.0, 2000.0
total_range = v_max - v_min

def get_pos(val):
    return (val - v_min) / total_range

# Build a non-linear color structure: Smooth gradients deep down, hard steps up high
# UPDATED: Hard boundaries are now shifted to -220 and -80 to match your Section 4 threshold
cdict = [
    (get_pos(-5000), "#000814"), # Deep ocean floor (starts smooth gradient)
    (get_pos(-1500), "#001d3d"), # Mid-depth gradient point
    (get_pos(-220),  "#003566"), # Shelf boundary lower cliff (gradient ends here)
    
    # Sharp step for the shelf flat zone
    (get_pos(-220),  "#00509d"), 
    (get_pos(-80),   "#00509d"), 
    
    # Sharp step for the coastline shallow water zone
    (get_pos(-80),   "#4cc9f0"), 
    (get_pos(0),     "#4cc9f0"), 
    
    # Sharp step for low land plains
    (get_pos(0),     "#a3b18a"), 
    (get_pos(500),   "#a3b18a"),
    
    # Sharp step for mountainous regions
    (get_pos(500),   "#588157"),
    (get_pos(2000),  "#344e41")
]

hybrid_cmap = LinearSegmentedColormap.from_list("hybrid_map", cdict, N=256)

# Plot using standard limits; our custom map inherently anchors the sharp edges
img = plt.imshow(marl_height_grid, origin='lower', extent=[0, width_km, 0, height_km], 
                 cmap=hybrid_cmap, vmin=v_min, vmax=v_max)

# Overlay the "-220 to -80" shelf region using your autumn/neon color scheme
plt.imshow(hotspot_overlay, origin='lower', extent=[0, width_km, 0, height_km], 
           cmap='autumn', alpha=0.6) 

# Add UI / Aesthetic Elements
plt.title(f"MARL Fishing Environment Map (Flattened Grid)\nResolution: {grid_res_km}km x {grid_res_km}km per cell", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("X Coordinate (Kilometers from West Bound)", fontsize=11)
plt.ylabel("Y Coordinate (Kilometers from South Bound)", fontsize=11)

# Depth Colorbar - Explicitly tick the new feature boundaries so you can verify alignment
ticks = [-5000, -3000, -1500, -220, -80, 0, 500, 2000]
cbar = plt.colorbar(img, orientation='vertical', shrink=0.75, pad=0.04, ticks=ticks)
cbar.set_label('Elevation / Depth (Meters)', rotation=270, labelpad=15)

plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
plt.tight_layout()

# Display the map
plt.show()
