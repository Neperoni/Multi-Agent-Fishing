# multiagent_fishing/simulation/Environment.py

import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple, List
from .vessels import PoacherVessel, PatrolVessel, SimulationVessel

class Environment:
    """
    Manages the pure mathematical state of the multi-agent fishing simulation.
    Operates strictly in a continuous kilometer vector space.
    Handles vessel lifecycles via unique integer IDs, spawning, and action routing.
    """
    def __init__(self, map_data, boat_names_path: str | Path = "datasets/boat_names.json"):
        """
        :param map_data: An instance of the Map class containing world dimensions and ports.
        :param boat_names_path: Path to the JSON file containing vessel naming components.
        """
        self.map_data = map_data
        
        # Core data registries keyed strictly on unique integer IDs
        self.poachers: Dict[int, PoacherVessel] = {}
        self.patrol_boats: Dict[int, PatrolVessel] = {}
        
        # Incremental ID tracking counter to assign the next available integer automatically
        self._next_id = 1

        # Track total elapsed simulation time in hours
        self.sim_time_hours = 0.0

        # Pre-load the structural name lists into memory once at startup
        self.name_pool = {"fishing_boats": {"first_word": [], "last_word": []},
                          "patrol_boats": {"first_word": [], "last_word": []}}
        
        n_path = Path(boat_names_path)
        if n_path.exists():
            with open(n_path, "r", encoding="utf-8") as f:
                self.name_pool = json.load(f)

    def _get_and_increment_id(self) -> int:
        """Finds and claims the next true sequential unique integer ID available."""
        while self._is_id_conflicting(self._next_id):
            self._next_id += 1
        claimed_id = self._next_id
        self._next_id += 1
        return claimed_id

    def add_poacher(self, vessel: PoacherVessel) -> bool:
        if self._is_id_conflicting(vessel.id):
            return False
        self.poachers[vessel.id] = vessel
        return True

    def add_patrol(self, vessel: PatrolVessel) -> bool:
        if self._is_id_conflicting(vessel.id):
            return False
        self.patrol_boats[vessel.id] = vessel
        return True

    def remove_vessel(self, vessel_id: int) -> bool:
        if vessel_id in self.poachers:
            del self.poachers[vessel_id]
            return True
        if vessel_id in self.patrol_boats:
            del self.patrol_boats[vessel_id]
            return True
        return False

    def _is_id_conflicting(self, vessel_id: int) -> bool:
        return (vessel_id in self.poachers) or (vessel_id in self.patrol_boats)

    @property
    def all_vessels(self) -> List[SimulationVessel]:
        return list(self.poachers.values()) + list(self.patrol_boats.values())

    def step(self, actions: Dict[int, Tuple[float, float]], dt: float = 0.25) -> dict:
        logs = {"out_of_bounds": [], "groundings": [], "apprehensions": []}
        # Advance the simulation internal clock by the timestep duration
        self.sim_time_hours += dt
        
        for vessel_id, vessel in self.poachers.items():
            if vessel_id in actions and vessel.is_active:
                heading, speed = actions[vessel_id]
                vessel.apply_movement(heading, speed, dt)
                self._enforce_environmental_constraints(vessel, logs)

        for vessel_id, vessel in self.patrol_boats.items():
            if vessel_id in actions and vessel.is_active:
                heading, speed = actions[vessel_id]
                vessel.apply_movement(heading, speed, dt)
                self._enforce_environmental_constraints(vessel, logs)

        self._check_patrol_interceptions(logs)
        return logs

    def _enforce_environmental_constraints(self, vessel: SimulationVessel, logs: dict):
        if not self.map_data.is_in_bounds(vessel.x, vessel.y):
            vessel.is_active = False
            vessel.current_speed = 0.0
            logs["out_of_bounds"].append(vessel.id)
            return

        elevation = self.map_data.get_height_at_pixel(vessel.x, vessel.y, method="closest")
        if elevation > 0.0:
            vessel.is_active = False
            vessel.current_speed = 0.0
            logs["groundings"].append(vessel.id)

    def _check_patrol_interceptions(self, logs: dict, catch_radius_km: float = 2.0):
        for patrol_id, patrol in self.patrol_boats.items():
            if not patrol.is_active:
                continue
            for poacher_id, poacher in self.poachers.items():
                if not poacher.is_active or poacher.is_caught:
                    continue
                distance = math.hypot(patrol.x - poacher.x, patrol.y - poacher.y)
                if distance <= catch_radius_km:
                    poacher.is_caught = True
                    poacher.is_active = False
                    poacher.current_speed = 0.0
                    logs["apprehensions"].append({"poacher_id": poacher_id, "by_patrol_id": patrol_id})
