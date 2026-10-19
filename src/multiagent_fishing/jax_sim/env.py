import jax
import jax.numpy as jnp
from typing import NamedTuple

from .maps import (GridMap, sample_height, sample_biomass, in_bounds,
                   sample_water_positions)
from .policy import TeamDims, POACHER_ACTIONS, PATROL_ACTIONS

POACHER = "poacher"
PATROL = "patrol"


class EnvConfig(NamedTuple):
    num_poachers: int = 8
    num_patrols: int = 3
    dt: float = 0.25                  # hours per step
    max_steps: int = 1000

    poacher_max_speed: float = 20.0   # km/h
    poacher_fishing_speed: float = 3.0
    poacher_max_cargo: float = 50.0
    patrol_max_speed: float = 45.0    # km/h
    catch_radius_km: float = 2.0

    # Reward shaping. These are placeholders to tune.
    reward_cargo: float = 1.0
    reward_caught: float = 10.0
    reward_apprehension: float = 10.0


class PoacherState(NamedTuple):
    x: jax.Array          # (N,) km
    y: jax.Array
    heading: jax.Array    # radians, 0 = north, clockwise
    speed: jax.Array      # km/h, last applied
    active: jax.Array     # (N,) bool
    cargo: jax.Array      # (N,)
    fishing: jax.Array    # (N,) bool
    caught: jax.Array     # (N,) bool


class PatrolState(NamedTuple):
    x: jax.Array          # (M,)
    y: jax.Array
    heading: jax.Array
    speed: jax.Array
    active: jax.Array


class EnvState(NamedTuple):
    t: jax.Array          # int32 scalar
    poachers: PoacherState
    patrols: PatrolState


def _wrap(h):
    return jnp.mod(h, 2 * jnp.pi)


class FishingEnv:
    """Pure-functional multi-team fishing environment.

    Methods take and return arrays only, so they are jit/vmap friendly.
    Wrap them with jax.jit / jax.vmap at the call site.
    """

    def __init__(self, cfg: EnvConfig, grid: GridMap):
        self.cfg = cfg
        self.grid = grid
        H, W = grid.height.shape
        width_km = W * float(grid.cell_km)
        height_km = H * float(grid.cell_km)
        self.scale = max(width_km, height_km)        # normalizes positions
        self.cx = float(grid.x0) + width_km / 2
        self.cy = float(grid.y0) + height_km / 2

    # ------------------------------------------------------------------ reset

    def reset(self, key) -> EnvState:
        cfg, g = self.cfg, self.grid
        N, M = cfg.num_poachers, cfg.num_patrols
        kp, kb, kph, kbh = jax.random.split(key, 4)

        px, py = sample_water_positions(kp, g, N)
        bx, by = sample_water_positions(kb, g, M)
        ph = jax.random.uniform(kph, (N,), minval=0.0, maxval=2 * jnp.pi)
        bh = jax.random.uniform(kbh, (M,), minval=0.0, maxval=2 * jnp.pi)

        poachers = PoacherState(
            x=px, y=py, heading=ph, speed=jnp.zeros(N),
            active=jnp.ones(N, bool), cargo=jnp.zeros(N),
            fishing=jnp.zeros(N, bool), caught=jnp.zeros(N, bool),
        )
        patrols = PatrolState(
            x=bx, y=by, heading=bh, speed=jnp.zeros(M), active=jnp.ones(M, bool),
        )
        return EnvState(t=jnp.zeros((), jnp.int32), poachers=poachers, patrols=patrols)

    # ------------------------------------------------------------------- step

    def step(self, s: EnvState, actions: dict):
        """
        actions[POACHER]: (N, 3) = [heading_rad, speed_frac, fish_logit_or_0_1]
        actions[PATROL]:  (M, 2) = [heading_rad, speed_frac]
        Speed is clipped to [0, 1]; the fish flag is thresholded at 0.5.
        Returns (next_state, rewards, done, info), where rewards[team] is per-agent.
        """
        cfg, g, dt = self.cfg, self.grid, self.cfg.dt
        p, q = s.poachers, s.patrols

        # ---- poachers: movement
        a = actions[POACHER]
        fishing = jnp.where(p.active, a[:, 2] > 0.5, p.fishing)
        p_heading = jnp.where(p.active, _wrap(a[:, 0]), p.heading)
        vmax = jnp.where(fishing, cfg.poacher_fishing_speed, cfg.poacher_max_speed)
        p_speed = jnp.where(p.active, jnp.clip(a[:, 1], 0.0, 1.0) * vmax, 0.0)
        px = p.x + jnp.sin(p_heading) * p_speed * dt
        py = p.y + jnp.cos(p_heading) * p_speed * dt

        # ---- poachers: environmental constraints
        p_off = (~in_bounds(g, px, py)) | (sample_height(g, px, py) > 0.0)
        p_active = p.active & ~p_off
        p_speed = jnp.where(p_active, p_speed, 0.0)

        # ---- poachers: cargo extraction
        biomass = sample_biomass(g, px, py)
        harvest = p_active & fishing & (p.cargo < cfg.poacher_max_cargo)
        cargo = jnp.where(
            harvest,
            jnp.minimum(cfg.poacher_max_cargo, p.cargo + biomass * dt),
            p.cargo,
        )
        delta_cargo = cargo - p.cargo

        # ---- patrols: movement
        b = actions[PATROL]
        q_heading = jnp.where(q.active, _wrap(b[:, 0]), q.heading)
        q_speed = jnp.where(q.active, jnp.clip(b[:, 1], 0.0, 1.0) * cfg.patrol_max_speed, 0.0)
        qx = q.x + jnp.sin(q_heading) * q_speed * dt
        qy = q.y + jnp.cos(q_heading) * q_speed * dt

        q_off = (~in_bounds(g, qx, qy)) | (sample_height(g, qx, qy) > 0.0)
        q_active = q.active & ~q_off
        q_speed = jnp.where(q_active, q_speed, 0.0)

        # ---- interceptions (N x M distance matrix)
        dist = jnp.hypot(px[:, None] - qx[None, :], py[:, None] - qy[None, :])
        in_range = (dist <= cfg.catch_radius_km) & p_active[:, None] & q_active[None, :]
        caught_now = in_range.any(axis=1)
        by_patrol = jnp.argmax(in_range.astype(jnp.int32), axis=1)
        p_active = p_active & ~caught_now
        p_speed = jnp.where(p_active, p_speed, 0.0)
        caught = p.caught | caught_now

        # ---- rewards (per agent)
        r_poacher = cfg.reward_cargo * delta_cargo - cfg.reward_caught * caught_now
        counts = jnp.zeros(cfg.num_patrols).at[by_patrol].add(caught_now.astype(jnp.float32))
        r_patrol = cfg.reward_apprehension * counts

        t = s.t + 1
        done = (t >= cfg.max_steps) | ~p_active.any()

        new_state = EnvState(
            t=t,
            poachers=PoacherState(x=px, y=py, heading=p_heading, speed=p_speed,
                                  active=p_active, cargo=cargo, fishing=fishing, caught=caught),
            patrols=PatrolState(x=qx, y=qy, heading=q_heading, speed=q_speed, active=q_active),
        )
        rewards = {POACHER: r_poacher, PATROL: r_patrol}
        info = {"caught": caught_now, "harvested": delta_cargo, "by_patrol": by_patrol}
        return new_state, rewards, done, info

    def step_with_reset(self, s: EnvState, actions: dict, key):
        """Auto-reset variant. The returned state is fresh when done is True.
        Note: the observation for the terminal transition is lost, so algorithms
        that bootstrap from the final state need the terminal observation stored
        before this call."""
        s_next, rewards, done, info = self.step(s, actions)
        fresh = self.reset(key)
        s_out = jax.tree_util.tree_map(lambda f, n: jnp.where(done, f, n), fresh, s_next)
        return s_out, rewards, done, info

    # ------------------------------------------------------------ observations

    def _norm_xy(self, x, y):
        return (x - self.cx) / self.scale, (y - self.cy) / self.scale

    def _self_features(self, s: EnvState):
        cfg, g = self.cfg, self.grid
        p, q = s.poachers, s.patrols
        f32 = lambda a: a.astype(jnp.float32)
        px, py = self._norm_xy(p.x, p.y)
        qx, qy = self._norm_xy(q.x, q.y)
        p_feat = jnp.stack([
            px, py, jnp.sin(p.heading), jnp.cos(p.heading),
            p.speed / cfg.poacher_max_speed, f32(p.active),
            p.cargo / cfg.poacher_max_cargo, f32(p.fishing),
            sample_biomass(g, p.x, p.y),
        ], axis=-1)                                           # (N, 9)
        q_feat = jnp.stack([
            qx, qy, jnp.sin(q.heading), jnp.cos(q.heading),
            q.speed / cfg.patrol_max_speed, f32(q.active),
            sample_biomass(g, q.x, q.y),
        ], axis=-1)                                           # (M, 7)
        return p_feat, q_feat

    def _relative(self, sx, sy, tx, ty, t_visible, self_eye=False):
        """Relative features from K source agents to L target entities: (K, L*3)."""
        dx = (tx[None, :] - sx[:, None]) / self.scale
        dy = (ty[None, :] - sy[:, None]) / self.scale
        vis = jnp.broadcast_to(t_visible[None, :].astype(jnp.float32), dx.shape)
        if self_eye:
            vis = vis * (1.0 - jnp.eye(dx.shape[0], dx.shape[1]))
        feats = jnp.stack([dx * vis, dy * vis, vis], axis=-1)  # (K, L, 3)
        return feats.reshape(dx.shape[0], -1)

    def observe(self, s: EnvState) -> dict:
        """Local observation for each agent, per team: {team: (num_agents, obs_dim)}."""
        p, q = s.poachers, s.patrols
        p_feat, q_feat = self._self_features(s)
        obs_p = jnp.concatenate([
            p_feat,
            self._relative(p.x, p.y, p.x, p.y, p.active, self_eye=True),
            self._relative(p.x, p.y, q.x, q.y, q.active),
        ], axis=-1)
        obs_b = jnp.concatenate([
            q_feat,
            self._relative(q.x, q.y, p.x, p.y, p.active),
            self._relative(q.x, q.y, q.x, q.y, q.active, self_eye=True),
        ], axis=-1)
        return {POACHER: obs_p, PATROL: obs_b}

    def global_state(self, s: EnvState):
        """Flat state vector used by centralized critics (MAPPO-style)."""
        p_feat, q_feat = self._self_features(s)
        return jnp.concatenate([p_feat.reshape(-1), q_feat.reshape(-1)])

    def critic_inputs(self, s: EnvState, critic_mode: str) -> dict:
        """'local' gives each team its own observations (IPPO / A2C).
        'global' gives every agent the global state (MAPPO)."""
        if critic_mode == "local":
            return self.observe(s)
        if critic_mode == "global":
            g = self.global_state(s)
            obs = self.observe(s)
            return {name: jnp.broadcast_to(g, (o.shape[0], g.shape[0]))
                    for name, o in obs.items()}
        raise ValueError(f"unknown critic_mode {critic_mode!r}")

    def dims(self, critic_mode: str) -> dict:
        """Shapes needed to build networks. Probed from the real functions, so it can't drift."""
        def _probe(key):
            s = self.reset(key)
            return self.observe(s), self.global_state(s)

        obs, g = jax.eval_shape(_probe, jax.random.PRNGKey(0))
        specs = {POACHER: POACHER_ACTIONS, PATROL: PATROL_ACTIONS}
        out = {}
        for name in (POACHER, PATROL):
            obs_dim = obs[name].shape[-1]
            critic_dim = obs_dim if critic_mode == "local" else g.shape[-1]
            out[name] = TeamDims(obs_dim, critic_dim, specs[name])
        return out