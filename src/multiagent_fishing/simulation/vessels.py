# multiagent_fishing/simulation/vessels.py
from enum import Enum
import numpy as np

class Role(Enum):
    POACHER = "poacher"
    PATROL = "patrol"

class SimulationVessel:
    def __init__(self, name: str, max_speed_kmh: float, start_x: float = 0.0, start_y: float = 0.0, start_heading: float = 0.0):
        self.id = None                  # Will be assigned automatically by the Environment
        self.name = name
        self.max_speed = max_speed_kmh  # Speed limit in km/h
        self.x = float(start_x)         # Continuous world X position in kilometers
        self.y = float(start_y)         # Continuous world Y position in kilometers
        self.current_heading = float(start_heading)  # Radians; 0 = North, increasing clockwise
        self.current_speed = 0.0
        self.is_active = True           # False if caught, sunk, or crashed
        self.role: Role = None  # set by subclasses

    def apply_movement(self, heading_rad, speed_fraction, dt=0.25):
        if not self.is_active:
            return
            
        actual_speed = np.clip(speed_fraction, 0.0, 1.0) * self.max_speed
        self.current_heading = heading_rad
        self.current_speed = actual_speed
        
        dx_km = np.sin(heading_rad) * actual_speed * dt
        dy_km = np.cos(heading_rad) * actual_speed * dt
        
        self.x += dx_km
        self.y += dy_km


# multiagent_fishing/simulation/vessels.py

class PoacherVessel(SimulationVessel):
    def __init__(self, name: str, start_x: float, start_y: float, home_port_id, start_heading: float = 0.0):
        super().__init__(name, max_speed_kmh=20.0, start_x=start_x, start_y=start_y, start_heading=start_heading)
        self.cargo = 0.0          
        self.max_cargo = 50.0     
        self.is_caught = False    
        self.home_x = start_x
        self.home_y = start_y
        self.home_port_id = home_port_id

        # New discrete state variables for RL tracking
        self.is_fishing = False
        self.FISHING_SPEED_LIMIT = 3.0 #speed cap while nets are deployed
        self.role = Role.POACHER

    def apply_movement(self, heading_rad, speed_fraction, dt=0.25):
        if not self.is_active:
            return
            
        # Dynamically set the speed ceiling based on the discrete fishing mode toggle
        allowed_max_speed = self.FISHING_SPEED_LIMIT if self.is_fishing else self.max_speed
        
        actual_speed = np.clip(speed_fraction, 0.0, 1.0) * allowed_max_speed
        self.current_heading = heading_rad
        self.current_speed = actual_speed
        
        dx_km = np.sin(heading_rad) * actual_speed * dt
        dy_km = np.cos(heading_rad) * actual_speed * dt
        
        self.x += dx_km
        self.y += dy_km


class PatrolVessel(SimulationVessel):
    def __init__(self, name: str, start_x: float, start_y: float, start_heading: float = 0.0):
        super().__init__(name, max_speed_kmh=45.0, start_x=start_x, start_y=start_y, start_heading=start_heading)
        self.role=Role.PATROL