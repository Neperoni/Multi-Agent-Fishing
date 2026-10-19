import jax
import jax.numpy as jnp

from .env import FishingEnv, POACHER, PATROL
from .policy import sample_action, log_prob

TEAMS = (POACHER, PATROL)


def make_rollout_fn(env: FishingEnv, nets: dict, num_envs: int, horizon: int,
                    critic_mode: str = "local"):
    """Returns jitted rollout(params, key) -> trajectory.
    Each leaf of the trajectory has shape (horizon, num_envs, ...)."""

    def rollout(params, key):
        k_init, k_scan = jax.random.split(key)
        states = jax.vmap(env.reset)(jax.random.split(k_init, num_envs))

        def one_step(states, key_t):
            k_act, k_env = jax.random.split(key_t)
            obs = jax.vmap(env.observe)(states)
            crit = jax.vmap(lambda s: env.critic_inputs(s, critic_mode))(states)

            actions, logps, values = {}, {}, {}
            for name, k in zip(TEAMS, jax.random.split(k_act, len(TEAMS))):
                dist = nets[name].apply(params[name], obs[name], method="actor")
                a = sample_action(dist, k)
                actions[name] = a
                logps[name] = log_prob(dist, a)
                values[name] = nets[name].apply(params[name], crit[name], method="value")

            env_keys = jax.random.split(k_env, num_envs)
            states, rewards, done, _ = jax.vmap(env.step_with_reset)(states, actions, env_keys)

            transition = {
                "obs": obs, "critic_obs": crit, "actions": actions,
                "log_probs": logps, "values": values,
                "rewards": rewards, "done": done,
            }
            return states, transition

        _, traj = jax.lax.scan(one_step, states, jax.random.split(k_scan, horizon))
        return traj

    return jax.jit(rollout)