# src/config.py

# --- Spatial Constants ---
GRID_RESOLUTION_METERS = 500.0

# How many pixels represent 1 Kilometer? 
# (e.g., 1000m / 500m = 2.0 pixels per km)
PIXELS_PER_KM = 1000.0 / GRID_RESOLUTION_METERS

# --- Temporal Constants ---
TIME_STEP_HOURS = 0.25  # 15 minutes per simulation tick
