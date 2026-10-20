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
        
        def _where_done(done, fresh, nxt):
            def pick(f, n):
                d = done.reshape(done.shape + (1,) * (n.ndim - done.ndim))
                return jnp.where(d, f, n)
            return jax.tree_util.tree_map(pick, fresh, nxt)

        # inside make_rollout_fn, replace one_step's tail with:
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
            s_next, rewards, done, _ = jax.vmap(env.step)(states, actions)
            fresh = jax.vmap(env.reset)(env_keys)
            next_states = _where_done(done, fresh, s_next)

            transition = {
                "obs": obs, "critic_obs": crit, "actions": actions,
                "log_probs": logps, "values": values,
                "rewards": rewards, "done": done,
                "state": states,         # state at the start of this step (every frame)
                "next_state": s_next,    # state after this step, before any reset
            }
            return next_states, transition

        _, traj = jax.lax.scan(one_step, states, jax.random.split(k_scan, horizon))
        return traj

    return jax.jit(rollout)