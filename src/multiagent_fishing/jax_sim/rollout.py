import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

from multiagent_fishing.brains.abstract_brain import AbstractBrain
from .env import FishingEnv, EnvState, POACHER, PATROL

TEAMS = (POACHER, PATROL)

def make_rollout_fn(env: FishingEnv, nets: Dict[str, AbstractBrain], num_envs: int, horizon: int):
    """Returns a jitted execution rollout function.
    
    Each leaf of the returning trajectory dictionary spans a clear shape matrix
    structured as: (horizon, num_envs, ...).
    """

    def rollout(params: dict, key: jax.Array) -> dict:
        k_init, k_scan = jax.random.split(key)
        
        # 1. Initialize parallel vectorized environment states across device memory
        states = jax.vmap(env.reset)(jax.random.split(k_init, num_envs))
        
        def _where_done(done: jax.Array, fresh: EnvState, nxt: EnvState) -> EnvState:
            """Applies masking criteria to reset terminated environments automatically."""
            def pick(f, n):
                # Dynamically broadcast the 'done' array shape across higher dimensional arrays
                d = done.reshape(done.shape + (1,) * (n.ndim - done.ndim))
                return jnp.where(d, f, n)
            return jax.tree_util.tree_map(pick, fresh, nxt)

        def one_step(states_t: EnvState, key_t: jax.Array) -> Tuple[EnvState, dict]:
            k_act, k_env = jax.random.split(key_t)

            actions, log_probs = {}, {}
            
            # 2. Query each swappable brain team using parallel env dimensions
            for name, k_team in zip(TEAMS, jax.random.split(k_act, len(TEAMS))):
                brain = nets[name]
                
                # Use jax.vmap to run the brain model over the entire batch of running environments (num_envs)
                batched_brain_call = jax.vmap(brain.get_action_jax, in_axes=(None, 0, None, 0))
                team_keys = jax.random.split(k_team, num_envs)
                
                # The brain processes raw simulation data directly
                brain_out = batched_brain_call(params[name], states_t, env.cfg, team_keys)
                
                actions[name] = brain_out.action
                log_probs[name] = brain_out.log_prob

            # 3. Advance parallelized environment physics sandbox steps
            env_keys = jax.random.split(k_env, num_envs)
            s_next, rewards, done, _ = jax.vmap(env.step)(states_t, actions)
            
            # 4. Extract auto-reset baselines for environments returning done flags
            fresh = jax.vmap(env.reset)(env_keys)
            next_states = _where_done(done, fresh, s_next)

            # 5. Pack everything directly into the step's transition output data buffer
            transition = {
                "state": states_t,      # State at the start of this step
                "next_state": s_next,   # Final state right before auto-resetting
                "actions": actions,
                "log_probs": log_probs, 
                "rewards": rewards, 
                "done": done,
            }
            return next_states, transition

        # Scan efficiently through the temporal execution axis
        _, traj = jax.lax.scan(one_step, states, jax.random.split(k_scan, horizon))
        return traj

    return jax.jit(rollout)
