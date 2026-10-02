# src/environment.py
import rasterio
import numpy as np
from pathlib import Path
from ..config import PIXELS_PER_KM, TIME_STEP_HOURS
from .vessels import PoacherVessel, PatrolVessel
import xarray as xr

class MaritimeSimEnvironment:
    """The core engine that holds data arrays, validates collisions, and runs a single frame step."""
    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Height map file not found at: {file_path}")
            
        ds = xr.open_dataset(file_path)
        lon_min = float(ds.lon.min())
        lon_max = float(ds.lon.max())
        lat_min = float(ds.lat.min())
        lat_max = float(ds.lat.max())






    def spawn_poacher(self, name: str, x: float, y: float):
        """Spawns and registers a poacher boat in the environment."""
        vessel = PoacherVessel(name, start_x=x, start_y=y)
        self.vessels[name] = vessel
        return vessel

    def spawn_patrol(self, name: str, x: float, y: float):
        """Spawns and registers a patrol boat in the environment."""
        vessel = PatrolVessel(name, start_x=x, start_y=y)
        self.vessels[name] = vessel
        return vessel

    def check_collisions(self, vessel):
        """Validates if a vessel hit land or went out of bounds."""
        px, py = int(vessel.x), int(vessel.y)
        
        # Check out-of-bounds boundary limits
        if not (0 <= px < self.height_map.shape[1] and 0 <= py < self.height_map.shape[0]):
            vessel.is_active = False
            return True
            
        # Read elevation (anything above 0m is land)
        if self.height_map[py, px] > 0:
            vessel.is_active = False
            return True
            
        return False

    def step(self, actions: dict):
        """Advances the simulation by 1 frame and returns raw object references."""
        for name, vessel in self.vessels.items():
            if not vessel.is_active:
                continue
                
            if name in actions:
                heading_rad, speed_fraction = actions[name]
                vessel.apply_movement(heading_rad, speed_fraction)
                self.check_collisions(vessel)
                
        poachers = [v for v in self.vessels.values() if isinstance(v, PoacherVessel)]
        patrols = [v for v in self.vessels.values() if isinstance(v, PatrolVessel)]
        
        return {"poachervessels":poachers, "patrolvessels":patrols}
    
    @staticmethod
    def print_vessel_table(vessel_data: dict):
        print(f"\n{'NAME':<18} | {'POS (X, Y)':<16} | {'HEADING':<9} | {'SPEED':<12} | {'CARGO':<9} | {'STATUS':<6}")
        print("-" * 83)
        
        for data in vessel_data["poachers"]:
            pos_str = f"({data['x']:.1f}, {data['y']:.1f})"
            heading_deg = np.degrees(data["heading"])
            speed_str = f"{data['speed']:.1f} km/h"
            cargo_str = f"{data['cargo']:.1f} t"
            alive_str = "ALIVE" if data["active"] else "DEAD"
            print(f"{data['name']:<18} | {pos_str:<16} | {heading_deg:>5.1f}° | {speed_str:>12} | {cargo_str:>7} | {alive_str:<6}")
            
        for data in vessel_data["patrols"]:
            pos_str = f"({data['x']:.1f}, {data['y']:.1f})"
            heading_deg = np.degrees(data["heading"])
            speed_str = f"{data['speed']:.1f} km/h"
            alive_str = "ALIVE" if data["active"] else "DEAD"
            print(f"{data['name']:<18} | {pos_str:<16} | {heading_deg:>5.1f}° | {speed_str:>12} | {'N/A':>7} | {alive_str:<6}")
            
        print("-" * 83 + "\n")
