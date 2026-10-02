# multiagent_fishing/simulation/vessels.py

import numpy as np

class SimulationVessel:
    def __init__(self, vessel_id: int, name: str, max_speed_kmh: float, start_x: float = 0.0, start_y: float = 0.0, start_heading: float = 0.0):
        self.id = int(vessel_id)         # Unique tracking ID enforced as an integer
        self.name = name
        self.max_speed = max_speed_kmh  # Speed limit in km/h
        self.x = float(start_x)         # Continuous world X position in kilometers
        self.y = float(start_y)         # Continuous world Y position in kilometers
        self.current_heading = float(start_heading)  # Radians; 0 = North, increasing clockwise
        self.current_speed = 0.0
        self.is_active = True           # False if caught, sunk, or crashed

    def apply_movement(self, heading_rad, speed_fraction, dt=0.25):
        if not self.is_active:
            return
            
        actual_speed = np.clip(speed_fraction, 0.0, 1.0) * self.max_speed
        self.current_heading = heading_rad
        self.current_speed = actual_speed
        
        # Calculate distance traveled in kilometers during this timestep
        dx_km = np.sin(heading_rad) * actual_speed * dt
        dy_km = np.cos(heading_rad) * actual_speed * dt
        
        self.x += dx_km
        self.y += dy_km


# multiagent_fishing/simulation/vessels.py

class PoacherVessel(SimulationVessel):
    def __init__(self, vessel_id: int, name: str, start_x: float, start_y: float, start_heading: float = 0.0):
        super().__init__(vessel_id, name, max_speed_kmh=20.0, start_x=start_x, start_y=start_y, start_heading=start_heading)
        self.cargo = 0.0          
        self.max_cargo = 50.0     
        self.is_caught = False    
        
        # Track the exact coordinate values of the harbor where this boat was created
        self.home_x = start_x
        self.home_y = start_y


class PatrolVessel(SimulationVessel):
    def __init__(self, vessel_id: int, name: str, start_x: float, start_y: float, start_heading: float = 0.0):
        super().__init__(vessel_id, name, max_speed_kmh=45.0, start_x=start_x, start_y=start_y, start_heading=start_heading)
