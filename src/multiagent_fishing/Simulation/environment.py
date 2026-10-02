# multiagent_fishing/simulation/Environment.py

import math
from typing import Dict, Tuple, List
from .vessels import PoacherVessel, PatrolVessel, SimulationVessel

class Environment:
    """
    Manages the pure mathematical state of the multi-agent fishing simulation.
    Operates strictly in a continuous kilometer vector space.
    Handles vessel lifecycles via unique integer IDs and action routing.
    """
    def __init__(self, map_data):
        """
        :param map_data: An instance of the Map class containing world dimensions.
        """
        self.map_data = map_data
        
        # Core data registries keyed strictly on unique integer IDs
        self.poachers: Dict[int, PoacherVessel] = {}
        self.patrol_boats: Dict[int, PatrolVessel] = {}

    def add_poacher(self, vessel: PoacherVessel) -> bool:
        """Exposes and registers a pre-instantiated poacher object if its integer ID is completely unique."""
        if self._is_id_conflicting(vessel.id):
            print(f"Simulation Error: Cannot add poacher. ID {vessel.id} is already registered.")
            return False
            
        self.poachers[vessel.id] = vessel
        return True

    def add_patrol(self, vessel: PatrolVessel) -> bool:
        """Exposes and registers a pre-instantiated patrol object if its integer ID is completely unique."""
        if self._is_id_conflicting(vessel.id):
            print(f"Simulation Error: Cannot add patrol. ID {vessel.id} is already registered.")
            return False
            
        self.patrol_boats[vessel.id] = vessel
        return True

    def remove_vessel(self, vessel_id: int) -> bool:
        """Removes a vessel and clears its integer ID tracking from the simulation entirely."""
        if vessel_id in self.poachers:
            del self.poachers[vessel_id]
            return True
        if vessel_id in self.patrol_boats:
            del self.patrol_boats[vessel_id]
            return True
        return False

    def _is_id_conflicting(self, vessel_id: int) -> bool:
        """Ensures integer IDs remain completely unique across all fleets."""
        return (vessel_id in self.poachers) or (vessel_id in self.patrol_boats)

    @property
    def all_vessels(self) -> List[SimulationVessel]:
        """Returns a flat list of all tracked vessels (directly consumable by BoatView loops)."""
        return list(self.poachers.values()) + list(self.patrol_boats.values())

    def step(self, actions: Dict[int, Tuple[float, float]], dt: float = 0.25) -> dict:
        """
        Advances the entire continuous coordinate physics model forward by one simulation timestep.
        
        :param actions: Dict mapping vessel_id integers to a tuple of (heading_radians, speed_fraction)
        :param dt: Timestep duration in hours (e.g., 0.25 = 15 minutes)
        :return: A status dictionary detailing frame events (like apprehensions and groundings)
        """
        logs = {"out_of_bounds": [], "groundings": [], "apprehensions": []}
        
        # 1. Process continuous updates for Poachers
        for vessel_id, vessel in self.poachers.items():
            if vessel_id in actions and vessel.is_active:
                heading, speed = actions[vessel_id]
                vessel.apply_movement(heading, speed, dt)
                self._enforce_environmental_constraints(vessel, logs)

        # 2. Process continuous updates for Patrols
        for vessel_id, vessel in self.patrol_boats.items():
            if vessel_id in actions and vessel.is_active:
                heading, speed = actions[vessel_id]
                vessel.apply_movement(heading, speed, dt)
                self._enforce_environmental_constraints(vessel, logs)

        # 3. Handle intercept distance evaluations between fleets
        self._check_patrol_interceptions(logs)

        return logs

    def _enforce_environmental_constraints(self, vessel: SimulationVessel, logs: dict):
        """Checks if a ship has driven completely off the map edge or run aground on land masses."""
        # 1. Continuous Boundary Check
        if not self.map_data.is_in_bounds(vessel.x, vessel.y):
            vessel.is_active = False
            vessel.current_speed = 0.0
            logs["out_of_bounds"].append(vessel.id)
            return

        # 2. Continuous Grounding Check (Uses closest pixel indexing for terrain checks)
        elevation = self.map_data.get_height_at_pixel(vessel.x, vessel.y, method="closest")
        if elevation > 0.0:
            vessel.is_active = False
            vessel.current_speed = 0.0
            logs["groundings"].append(vessel.id)

    def _check_patrol_interceptions(self, logs: dict, catch_radius_km: float = 2.0):
        """Evaluates distances between fleets to check if active patrols apprehend poachers."""
        for patrol_id, patrol in self.patrol_boats.items():
            if not patrol.is_active:
                continue
                
            for poacher_id, poacher in self.poachers.items():
                if not poacher.is_active or poacher.is_caught:
                    continue
                
                # Pure continuous distance check in kilometer tracking space
                distance = math.hypot(patrol.x - poacher.x, patrol.y - poacher.y)
                if distance <= catch_radius_km:
                    poacher.is_caught = True
                    poacher.is_active = False
                    poacher.current_speed = 0.0
                    logs["apprehensions"].append({"poacher_id": poacher_id, "by_patrol_id": patrol_id})
