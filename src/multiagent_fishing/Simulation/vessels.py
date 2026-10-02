import numpy as np
from .. import config 

class SimulationVessel:
    def __init__(self, name, max_speed_kmh, start_x=0.0, start_y=0.0):
        self.name = name
        self.max_speed = max_speed_kmh  # Speed limit in km/h
        self.x = float(start_x)         # Map grid X pixel coordinate
        self.y = float(start_y)         # Map grid Y pixel coordinate
        self.current_heading = 0
        self.current_speed = 0
        self.is_active = True           # False if caught, sunk, or crashed

    def apply_movement(self, heading_rad, speed_fraction, dt=0.25):
        if not self.is_active:
            return
            
        # Clamp speed fraction between 0.0 (stopped) and 1.0 (full throttle)
        actual_speed = np.clip(speed_fraction, 0.0, 1.0) * self.max_speed
        
        self.current_heading = heading_rad
        self.current_speed = actual_speed
        
        # Calculate distance traveled in kilometers during this timestep
        dx_km = np.cos(heading_rad) * actual_speed * dt
        dy_km = np.sin(heading_rad) * actual_speed * dt
        
        # Convert km to 500m grid pixels (1 km = 2 pixels)
        self.x += dx_km * config.PIXELS_PER_KM
        self.y += dy_km * config.PIXELS_PER_KM

class PoacherVessel(SimulationVessel):
    def __init__(self, name, start_x, start_y):
        # Poachers travel at roughly 20 km/h max speed
        super().__init__(name, max_speed_kmh=20.0, start_x=start_x, start_y=start_y)
        
        self.cargo = 0.0          # Tons of fish currently caught
        self.max_cargo = 50.0     # Maximum hold capacity
        self.is_caught = False    # True if intercepted by DFO patrol

class PatrolVessel(SimulationVessel):
    def __init__(self, name, start_x, start_y):
        # Patrols are fast (45 km/h max speed)
        super().__init__(name, max_speed_kmh=45.0, start_x=start_x, start_y=start_y)