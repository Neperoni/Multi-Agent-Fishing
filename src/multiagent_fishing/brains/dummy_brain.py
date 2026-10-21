import jax
import jax.numpy as jnp
from typing import Any

from multiagent_fishing.brains.abstract_brain import AbstractBrain, BrainOutput
from multiagent_fishing.simulation.observation import Observation
from multiagent_fishing.jax_sim.env import EnvState, EnvConfig

# ----------------------------------------------------------------------------
# Dummy Poacher Brain
# ----------------------------------------------------------------------------
class DummyPoacherBrain(AbstractBrain):
    def get_action(self, obs: Observation):
        vessel = obs.vessel_state
        heading = vessel.current_heading

        # Drives in a straight line until it finds fish, then it catches fish
        fish_toggle = 1 if vessel.cargo < vessel.max_cargo else 0
        return (heading, 0.5, fish_toggle)

    def get_action_jax(self, params: Any, state: EnvState, cfg: EnvConfig, rng: jax.Array) -> BrainOutput:
        """Completed JAX implementation matching the linear dummy poacher strategy."""
        p = state.poachers
        num_agents = p.x.shape[0]

        # 1. Maintain current heading vector across the batch
        heading = p.heading  # Shape: (N,)
        
        # 2. Maintain continuous half speed
        speed = jnp.full((num_agents,), 0.5)  # Shape: (N,)

        # 3. Vectorized condition: fish if cargo is below max capacity limit
        # jnp.where handles element-wise conditional mapping instantly on GPU/TPU
        fish_toggle = jnp.where(p.cargo < cfg.poacher_max_cargo, 1.0, 0.0)  # Shape: (N,)

        # 4. Assemble the raw physical action matrix expected by FishingEnv.step
        actions = jnp.stack([heading, speed, fish_toggle], axis=-1)  # Shape: (N, 3)

        # 5. Dummy log-probabilities for PPO structural pipeline alignment
        log_probs = jnp.zeros((num_agents,))  # Shape: (N,)

        return BrainOutput(action=actions, log_prob=log_probs)


# ----------------------------------------------------------------------------
# Dummy Patrol Brain
# ----------------------------------------------------------------------------
class DummyPatrolBrain(AbstractBrain):
    def get_action(self, obs: Observation):
        heading = obs.vessel_state.current_heading
        
        # Drives in a straight line at half speed
        return (heading, 0.5)

    def get_action_jax(self, params: Any, state: EnvState, cfg: EnvConfig, rng: jax.Array) -> BrainOutput:
        """Completed JAX implementation matching the linear dummy patrol strategy."""
        q = state.patrols
        num_agents = q.x.shape[0]

        # 1. Maintain current heading vector across the batch
        heading = q.heading  # Shape: (M,)
        
        # 2. Maintain continuous half speed
        speed = jnp.full((num_agents,), 0.5)  # Shape: (M,)

        # 3. Assemble the raw physical action matrix expected by FishingEnv.step
        actions = jnp.stack([heading, speed], axis=-1)  # Shape: (M, 2)

        # 4. Dummy log-probabilities for PPO structural pipeline alignment
        log_probs = jnp.zeros((num_agents,))  # Shape: (M,)

        return BrainOutput(action=actions, log_prob=log_probs)
