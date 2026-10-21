from abc import ABC, abstractmethod
from multiagent_fishing.simulation.observation import Observation

from typing import Any, NamedTuple
import jax
from ..jax_sim.env import EnvState, EnvConfig

class BrainOutput(NamedTuple):
    action: jax.Array     # Matrix sent to the simulator (e.g., shape N x 3 or M x 2)
    log_prob: jax.Array   # Trajectory tracking buffer for PPO loss updates

class AbstractBrain(ABC):
    @abstractmethod
    def get_action(self, obs: Any):
        """Return the next action tuple for a traditional non-JAX vessel setup."""
        pass

    @abstractmethod
    def get_action_jax(self, params: Any, state: EnvState, cfg: EnvConfig, rng: jax.Array) -> BrainOutput:
        """Pure-functional JAX execution method.
        
        Reads raw state data directly and outputs physical action vectors 
        and PPO tracking metrics.
        """
        pass



