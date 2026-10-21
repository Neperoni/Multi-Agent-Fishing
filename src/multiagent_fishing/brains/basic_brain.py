# multiagent_fishing/brains/basic_brains.py
import math
import random

from multiagent_fishing.brains.abstract_brain import AbstractBrain
from multiagent_fishing.simulation.observation import Observation
from multiagent_fishing.simulation.vessels import Role
from multiagent_fishing.world import Layers

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

# Angular offsets to try, smallest turn first: 0, +15, -15, +30, -30, ... 180
_OFFSETS = [0.0]
for _deg in range(15, 181, 15):
    _OFFSETS += [math.radians(_deg), -math.radians(_deg)]


def point_is_safe(map_data, x, y) -> bool:
    """True if the point is in bounds and is open water."""
    if not map_data.is_in_bounds(x, y):
        return False
    return float(map_data.sample_point((x, y))[Layers.HEIGHT]) <= 0.0


def path_is_safe(map_data, x, y, heading, distance_km, samples=5) -> bool:
    """Checks several points along a straight line from (x, y) along heading."""
    for i in range(1, samples + 1):
        d = distance_km * i / samples
        px = x + math.sin(heading) * d
        py = y + math.cos(heading) * d
        if not point_is_safe(map_data, px, py):
            return False
    return True


def bearing_to(x, y, tx, ty) -> float:
    """Heading in radians (0 = North, clockwise) from (x, y) to (tx, ty)."""
    return math.atan2(tx - x, ty - y)


def safe_heading(map_data, x, y, desired, lookahead_km):
    """Returns the safe heading closest to `desired`, or None if all are blocked."""
    for off in _OFFSETS:
        h = desired + off
        if path_is_safe(map_data, x, y, h, lookahead_km):
            return h
    return None


def port_heading(map_data, x, y, port_id):
    """
    Returns (heading_rad, path_distance_km) toward the port along water, using
    map_data.nav_to_port. Heading is None if there is no path or the position
    is out of bounds. Assumes world-frame (dir_x = east, dir_y = north).
    """
    dist, dx, dy = map_data.nav_to_port(port_id, (x, y))
    dist, dx, dy = float(dist), float(dx), float(dy)
    if math.hypot(dx, dy) < 1e-9:
        return None, dist
    return math.atan2(dx, dy), dist


# ---------------------------------------------------------------------------
# Brains
# ---------------------------------------------------------------------------

class BasicPoacherBrain(AbstractBrain):
    """
    - Not full: sample biomass in a ring around the ship and steer toward the
      richest safe direction. Deploy nets when local biomass is good enough
      (PoacherVessel caps speed while fishing).
    - Full: follow the port flow map home.
    Tunable via obs.params: fish_threshold, lookahead_km, home_lookahead_km,
    num_directions, home_radius_km.
    Requires vessel.home_port_id.
    """


    #follows fish gradient to try and find more fish
    #if it gets above some threshold it starts fishing
    #when its full it then returns to home port using port gradient

    def get_action(self, obs: Observation):
        vessel = obs.vessel_state
        map_data = obs.env.map_data
        p = obs.params

        fish_threshold = p.get("fish_threshold", 0.3)
        lookahead = p.get("lookahead_km", 10.0)
        home_lookahead = p.get("home_lookahead_km", 5.0)
        num_dirs = p.get("num_directions", 16)
        home_radius = p.get("home_radius_km", 1.0)
        return_fraction = p.get("return_fraction", 0.5)  # 1.0 = only return when full

        # --- Cargo full: follow the port flow map home, nets in ---
        if vessel.cargo >= vessel.max_cargo * return_fraction:
            desired, path_dist = port_heading(
                map_data, vessel.x, vessel.y, vessel.home_port_id
            )

            if path_dist <= home_radius:
                return (vessel.current_heading, 0.0, 0)  # docked

            if desired is None:
                # No water path (or out of bounds): straight line home as a last resort
                desired = bearing_to(vessel.x, vessel.y, vessel.home_x, vessel.home_y)

            heading = safe_heading(map_data, vessel.x, vessel.y, desired, home_lookahead)
            if heading is None:
                return (vessel.current_heading, 0.0, 0)
            return (heading, 1.0, 0)

        # --- Looking for fish ---
        here = float(map_data.sample_point((vessel.x, vessel.y))[Layers.BIOMASS])

        best_heading, best_score = None, -1.0
        for i in range(num_dirs):
            h = 2 * math.pi * i / num_dirs
            if not path_is_safe(map_data, vessel.x, vessel.y, h, lookahead):
                continue
            px = vessel.x + math.sin(h) * lookahead
            py = vessel.y + math.cos(h) * lookahead
            score = float(map_data.sample_point((px, py))[Layers.BIOMASS])
            # Small bonus for the current heading to reduce jitter
            diff = abs((h - vessel.current_heading + math.pi) % (2 * math.pi) - math.pi)
            if diff < math.pi / num_dirs:
                score += 0.02
            if score > best_score:
                best_heading, best_score = h, score

        # Boxed in on every side
        if best_heading is None:
            return (vessel.current_heading, 0.0, 0)

        if here >= fish_threshold:
            # Nets down. Only creep toward richer water if there is some nearby.
            if best_score > here:
                return (best_heading, 1.0, 1)
            return (vessel.current_heading, 0.0, 1)

        # Cruise toward the richest direction; if the field is flat, keep going straight.
        if best_score <= here:
            heading = safe_heading(
                map_data, vessel.x, vessel.y, vessel.current_heading, lookahead
            )
            if heading is None:
                heading = best_heading
        else:
            heading = best_heading
        return (heading, 1.0, 0)


class BasicPatrolBrain(AbstractBrain):
    """
    Patrol boats travel in a straight line at full speed.
    - If a poacher is detected in obs.visible_ships, they aggressively intercept it.
    - If land or out-of-bounds is detected ahead, they execute a physical 'bounce-off'
      reflection to choose a new valid heading and keep moving.
    """

    def get_action(self, obs: Observation):
        vessel = obs.vessel_state
        map_data = obs.env.map_data
        p = obs.params

        lookahead = p.get("lookahead_km", 20.0)
        
        # 1. --- RADAR INTERCEPTION LOGIC ---
        # Filter matching the specific Role Enum object structure
        visible_poachers = [
            s for s in obs.visible_ships 
            if s.role == Role.POACHER and getattr(s, "is_active", True) and not getattr(s, "is_caught", False)
        ]

        if visible_poachers:
            # Target the closest poacher from our local radar array
            closest_poacher = min(
                visible_poachers, 
                key=lambda s: math.hypot(s.x - vessel.x, s.y - vessel.y)
            )
            intercept_heading = bearing_to(vessel.x, vessel.y, closest_poacher.x, closest_poacher.y)
            
            # Reduce lookahead distance during active hot pursuit to allow shallow-water chases
            chase_lookahead = min(5.0, lookahead / 4.0)
            safe_intercept = safe_heading(map_data, vessel.x, vessel.y, intercept_heading, chase_lookahead)
            
            if safe_intercept is not None:
                return (safe_intercept, 1.0) # Full speed intercept split
                
        # 2. --- COLLISION & BOUNDARY REFLECTION (BOUNCE-OFF) ---
        if not path_is_safe(map_data, vessel.x, vessel.y, vessel.current_heading, lookahead):
            # Calculate a mirrored bounce-back reflection vector
            bounce_angle = (vessel.current_heading + math.pi) % (2 * math.pi)
            new_heading = safe_heading(map_data, vessel.x, vessel.y, bounce_angle, lookahead)
            
            if new_heading is not None:
                # Convert 10 degrees to radians and apply a random variance between -10° and +10°
                random_angle_variation = random.uniform(math.radians(-40), math.radians(40))
                final_heading = (new_heading + random_angle_variation) % (2 * math.pi)
                
                return (final_heading, 1.0)
            else:
                return (vessel.current_heading + 0.03, 0.1)  # Boxed-in state safeguard

        # 3. --- DEFAULT STATE: CRUISE STRAIGHT ---
        return (vessel.current_heading, 1.0)
