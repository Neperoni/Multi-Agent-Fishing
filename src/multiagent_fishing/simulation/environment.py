# multiagent_fishing/simulation/environment.py

import math
from typing import Dict, Tuple, List
from .vessels import PoacherVessel, PatrolVessel, SimulationVessel
from multiagent_fishing.world import Layers
from .observation import Observation

class Environment:
    """
    Manages the pure mathematical state of the multi-agent fishing simulation.
    Handles automatic sequential integer ID assignment on registration.
    """
    def __init__(self, map_data):
        self.map_data = map_data
        self.poachers: Dict[int, PoacherVessel] = {}
        self.patrol_boats: Dict[int, PatrolVessel] = {}
        
        self._next_id = 1
        self.sim_time_hours = 0.0

        self._registered_references = set() 
        
        self.poacher_brain = None
        self.patrol_brain = None

    def add_vessel(self, vessel: SimulationVessel) -> int:
        """
        Unified entry point to register any vessel type.
        Safeguards against adding the exact same memory reference twice.
        """
        is_already_added = any(vessel is existing for existing in self.all_vessels)
        if is_already_added:
            print(f"Simulation Warning: Ignored duplicate request for {vessel.name}.")
            return vessel.id

        # 1. Track the unique object memory reference
        self._registered_references.add(id(vessel))

        # 2. Assign the next available simulation integer ID
        vessel.id = self._get_next_id()
        
        # 3. Route to the correct fleet dictionary
        if isinstance(vessel, PoacherVessel):
            self.poachers[vessel.id] = vessel
        elif isinstance(vessel, PatrolVessel):
            self.patrol_boats[vessel.id] = vessel
        else:
            raise TypeError(f"Cannot add unknown vessel subclass type: {type(vessel)}")
            
        return vessel.id

    def remove_vessel(self, vessel_id: int) -> bool:
        """Removes a vessel and clears its tracking states entirely."""
        # Look up the vessel instance to safely prune its memory address token from our reference set
        vessel = self.poachers.get(vessel_id) or self.patrol_boats.get(vessel_id)
        if vessel:
            self._registered_references.discard(id(vessel))
            
        if vessel_id in self.poachers:
            del self.poachers[vessel_id]
            return True
        if vessel_id in self.patrol_boats:
            del self.patrol_boats[vessel_id]
            return True
        return False
    
    def _get_next_id(self) -> int:
        """Finds the next unassigned sequential integer ID token."""
        while (self._next_id in self.poachers) or (self._next_id in self.patrol_boats):
            self._next_id += 1
        assigned_id = self._next_id
        self._next_id += 1
        return assigned_id

    @property
    def all_vessels(self) -> List[SimulationVessel]:
        return list(self.poachers.values()) + list(self.patrol_boats.values())

    def set_poacher_brain(self, brain_class):
        """Sets the stateless brain strategy for all poacher vessels."""
        self.poacher_brain = brain_class
    
    def set_patrol_brain(self, brain_class):
        """Sets the stateless brain strategy for all patrol vessels."""
        self.patrol_brain = brain_class

    def step(self, actions: dict, dt: float = 0.25) -> dict:
        """
        Advances the simulation physics and environments constraints.
        :param actions: Dict mapping ID to a tuple:
                        - Poachers: (heading_rad, speed_fraction, fishing_toggle)
                        - Patrols:  (heading_rad, speed_fraction)
        """
        self.sim_time_hours += dt
        logs = {"out_of_bounds": [], "groundings": [], "apprehensions": []}
        
        
        if(self.poacher_brain is None):
            print("Poacher brain is not defined, please call set_poacher_brain before running")
        
        if(self.patrol_brain is None):
            print("Patrol brain is not defined, please call set_patrol_brain before running")
        
        
        
        # 1. Process Poachers
        for vessel_id, vessel in self.poachers.items():
            if vessel_id in actions and vessel.is_active:
                # Unpack the discrete fishing toggle (0 = cruise, 1 = deploy nets)
                heading, speed, fish_toggle = actions[vessel_id]
                
                # The agent retains full control over its fishing state toggle at all times
                vessel.is_fishing = bool(fish_toggle == 1)

                vessel.apply_movement(heading, speed, dt)
                self._enforce_environmental_constraints(vessel, logs)
                
                # --- CARGO EXTRACTION MATH ---
                if vessel.is_active and vessel.is_fishing and vessel.cargo < vessel.max_cargo:
                    # Sample the background biomass scalar (0.0 to 1.0+)
                    biomass_density = float(self.map_data.sample_point((vessel.x, vessel.y))[Layers.BIOMASS])
                                        
                    # Extraction rate scales dynamically by background concentration value
                    harvest_rate_per_hour = biomass_density
                    
                    # Accumulate fish up to the physical max cargo hold boundary limit
                    vessel.cargo = min(vessel.max_cargo, vessel.cargo + (harvest_rate_per_hour * dt))

        # 2. Process Patrols
        for vessel_id, vessel in self.patrol_boats.items():
            if vessel_id in actions and vessel.is_active:
                heading, speed = actions[vessel_id][:2] # Ensure safe slicing
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

        elevation = self.map_data.sample_point((vessel.x, vessel.y))[Layers.HEIGHT]
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


    def get_observation(self, vessel_id: int, dt: float, params: dict = None) -> Observation:
        """Builds the decision-time Observation for a single vessel."""
        vessel = self.poachers.get(vessel_id) or self.patrol_boats.get(vessel_id)
        if vessel is None:
            raise KeyError(f"No vessel registered with id {vessel_id}")
            
        # 1. Pull the radar range setting from params
        actual_params = params or {}
        radar_radius = actual_params.get("radar_radius_km", 50.0)
        
        # 2. Compute nearby visible ships using the helper logic
        visible = []
        for other in self.all_vessels:
            if not other.is_active or other.id == vessel_id:
                continue
            distance = math.hypot(other.x - vessel.x, other.y - vessel.y)
            if distance <= radar_radius:
                visible.append(other)

        # 3. Deliver packaged, decoupled observation data
        return Observation(
            vessel_id=vessel_id,
            vessel_state=vessel,
            timestep=dt,
            role=vessel.role,
            visible_ships=visible,
            env=self,
            params=actual_params,
        )


    def get_actions(self, dt: float) -> dict:
        """Asks each role's brain for an action for every active vessel."""
        if self.poacher_brain is None or self.patrol_brain is None:
            raise RuntimeError("Set both brains with set_poacher_brain / set_patrol_brain first")

        actions = {}
        for vid, vessel in self.poachers.items():
            if vessel.is_active:
                actions[vid] = self.poacher_brain.get_action(self.get_observation(vid, dt))
        for vid, vessel in self.patrol_boats.items():
            if vessel.is_active:
                actions[vid] = self.patrol_brain.get_action(self.get_observation(vid, dt))
        return actions
    
    def get_vessels_in_radius(self, center_x: float, center_y: float, radius_km: float, exclude_id: int = None) -> List[SimulationVessel]:
        """
        Finds all active vessels within a given kilometer radius of a coordinate.
        
        :param center_x: The X coordinate of the scanning agent.
        :param center_y: The Y coordinate of the scanning agent.
        :param radius_km: The maximum detection range.
        :param exclude_id: Optional ID of the scanning vessel to prevent self-detection.
        :return: A list of SimulationVessel objects within range.
        """
        detected_vessels = []
        
        for vessel in self.all_vessels:
            if not vessel.is_active or vessel.id == exclude_id:
                continue
                
            distance = math.hypot(vessel.x - center_x, vessel.y - center_y)
            if distance <= radius_km:
                detected_vessels.append(vessel)
                
        return detected_vessels
