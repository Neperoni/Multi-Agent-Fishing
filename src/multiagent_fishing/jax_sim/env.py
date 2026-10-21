import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict

# Explicitly import your JaxMap container and port spawner
from .maps import JaxMap, jax_sample_port_positions

# Maintain your exact simulation constants
POACHER = "poacher"
PATROL = "patrol"

class EnvConfig(NamedTuple):
    num_poachers: int = 8
    num_patrols: int = 3
    dt: float = 0.25
    max_steps: int = 1000
    poacher_max_speed: float = 20.0
    poacher_fishing_speed: float = 3.0
    poacher_max_cargo: float = 50.0
    patrol_max_speed: float = 45.0
    catch_radius_km: float = 2.0
    reward_cargo: float = 1.0
    reward_caught: float = 10.0
    reward_apprehension: float = 10.0

class PoacherState(NamedTuple):
    x: jax.Array; y: jax.Array; heading: jax.Array; speed: jax.Array
    active: jax.Array; cargo: jax.Array; fishing: jax.Array; caught: jax.Array

class PatrolState(NamedTuple):
    x: jax.Array; y: jax.Array; heading: jax.Array; speed: jax.Array; active: jax.Array

class EnvState(NamedTuple):
    t: jax.Array
    poachers: PoacherState
    patrols: PatrolState

def _wrap(h):
    return jnp.mod(h, 2 * jnp.pi)

class FishingEnv:
    """A pure physical simulation sandbox. 
    
    Contains no RL awareness, critic inputs, or relative perceptual logic.
    """
    # Renamed 'grid: Any' to 'map_data: JaxMap' for strict type clarity
    def __init__(self, cfg: EnvConfig, map_data: JaxMap):
        self.cfg = cfg
        self.map_data = map_data  # Stored explicitly as map_data
        
        # Pull map configurations directly from the static map properties
        width_km = map_data.width_km
        height_km = map_data.height_km
        self.scale = max(width_km, height_km)
        
        self.cx = width_km / 2
        self.cy = height_km / 2

    def reset(self, key: jax.Array) -> EnvState:
        cfg, g = self.cfg, self.map_data
        N, M = cfg.num_poachers, cfg.num_patrols
        kp, kb, kph, kbh = jax.random.split(key, 4)

        # Boats spawn directly at the pre-loaded static port nodes
        px, py = jax_sample_port_positions(kp, g, N)
        bx, by = jax_sample_port_positions(kb, g, M)
        
        ph = jax.random.uniform(kph, (N,), minval=0.0, maxval=2 * jnp.pi)
        bh = jax.random.uniform(kbh, (M,), minval=0.0, maxval=2 * jnp.pi)

        return EnvState(
            t=jnp.zeros((), jnp.int32),
            poachers=PoacherState(px, py, ph, jnp.zeros(N), jnp.ones(N, bool), jnp.zeros(N), jnp.zeros(N, bool), jnp.zeros(N, bool)),
            patrols=PatrolState(bx, by, bh, jnp.zeros(M), jnp.ones(M, bool))
        )

    def step(self, s: EnvState, actions: Dict[str, jax.Array]):
        cfg, g, dt = self.cfg, self.map_data, self.cfg.dt
        p, q = s.poachers, s.patrols

        # ---- Kinematics: Poachers
        a_p = actions[POACHER]
        fishing = jnp.where(p.active, a_p[:, 2] > 0.5, p.fishing)
        p_heading = jnp.where(p.active, _wrap(a_p[:, 0]), p.heading)
        vmax = jnp.where(fishing, cfg.poacher_fishing_speed, cfg.poacher_max_speed)
        p_speed = jnp.where(p.active, jnp.clip(a_p[:, 1], 0.0, 1.0) * vmax, 0.0)
        px = p.x + jnp.sin(p_heading) * p_speed * dt
        py = p.y + jnp.cos(p_heading) * p_speed * dt

        # Import functional methods from your refactored JAX-optimized maps module
        from .maps import _jax_cell_index, JaxLayers
        
        row_p, col_p, inb_p = _jax_cell_index(g, px, py)
        p_heights = g.layers[row_p, col_p, JaxLayers.HEIGHT]
        p_off = (~inb_p) | (p_heights > 0.0)
        p_active = p.active & ~p_off
        p_speed = jnp.where(p_active, p_speed, 0.0)

        # ---- Harvesting Rules
        p_biomass = g.layers[row_p, col_p, JaxLayers.BIOMASS]
        harvest = p_active & fishing & (p.cargo < cfg.poacher_max_cargo)
        cargo = jnp.where(harvest, jnp.minimum(cfg.poacher_max_cargo, p.cargo + p_biomass * dt), p.cargo)
        delta_cargo = cargo - p.cargo

        # ---- Kinematics: Patrols
        a_q = actions[PATROL]
        q_heading = jnp.where(q.active, _wrap(a_q[:, 0]), q.heading)
        q_speed = jnp.where(q.active, jnp.clip(a_q[:, 1], 0.0, 1.0) * cfg.patrol_max_speed, 0.0)
        qx = q.x + jnp.sin(q_heading) * q_speed * dt
        qy = q.y + jnp.cos(q_heading) * q_speed * dt

        row_q, col_q, inb_q = _jax_cell_index(g, qx, qy)
        q_heights = g.layers[row_q, col_q, JaxLayers.HEIGHT]
        q_off = (~inb_q) | (q_heights > 0.0)
        q_active = q.active & ~q_off
        q_speed = jnp.where(q_active, q_speed, 0.0)

        # ---- Proximity Arrest Rules
        dist = jnp.hypot(px[:, None] - qx[None, :], py[:, None] - qy[None, :])
        in_range = (dist <= cfg.catch_radius_km) & p_active[:, None] & q_active[None, :]
        caught_now = in_range.any(axis=1)
        by_patrol = jnp.argmax(in_range.astype(jnp.int32), axis=1)
        p_active = p_active & ~caught_now
        p_speed = jnp.where(p_active, p_speed, 0.0)
        caught = p.caught | caught_now

        # ---- Economy System (Rewards tracking remains native to physics outputs)
        r_poacher = cfg.reward_cargo * delta_cargo - cfg.reward_caught * caught_now
        counts = jnp.zeros(cfg.num_patrols).at[by_patrol].add(caught_now.astype(jnp.float32))
        r_patrol = cfg.reward_apprehension * counts

        t = s.t + 1
        done = (t >= cfg.max_steps) | ~p_active.any()

        new_state = EnvState(
            t=t,
            poachers=PoacherState(px, py, p_heading, p_speed, p_active, cargo, fishing, caught),
            patrols=PatrolState(qx, qy, q_heading, q_speed, q_active)
        )
        return new_state, {POACHER: r_poacher, PATROL: r_patrol}, done, {"caught": caught_now}

    def _norm_xy(self, x, y):
        return (x - self.cx) / self.scale, (y - self.cy) / self.scale

    def _self_features(self, s: EnvState):
        """Outputs bare coordinate data directly derived from active engine states."""
        from .maps import _jax_cell_index, JaxLayers
        p, q, g = s.poachers, s.patrols, self.map_data
        px, py = self._norm_xy(p.x, p.y)
        qx, qy = self._norm_xy(q.x, q.y)
        
        row_p, col_p, _ = _jax_cell_index(g, p.x, p.y)
        p_biomass = g.layers[row_p, col_p, JaxLayers.BIOMASS]
        
        row_q, col_q, _ = _jax_cell_index(g, q.x, q.y)
        q_biomass = g.layers[row_q, col_q, JaxLayers.BIOMASS]
        
        p_feat = jnp.stack([px, py, jnp.sin(p.heading), jnp.cos(p.heading), p.speed / self.cfg.poacher_max_speed, p.active.astype(jnp.float32), p.cargo / self.cfg.poacher_max_cargo, p.fishing.astype(jnp.float32), p_biomass], axis=-1)
        q_feat = jnp.stack([qx, qy, jnp.sin(q.heading), jnp.cos(q.heading), q.speed / self.cfg.patrol_max_speed, q.active.astype(jnp.float32), q_biomass], axis=-1)
        return p_feat, q_feat
