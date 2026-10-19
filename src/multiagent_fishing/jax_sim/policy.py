from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


# ----------------------------------------------------------------------------
# Action spaces and distributions (algorithm-agnostic)
# ----------------------------------------------------------------------------

class ActionSpec(NamedTuple):
    cont_dim: int       # number of Gaussian dimensions
    n_discrete: int     # number of Bernoulli dimensions, appended after continuous


POACHER_ACTIONS = ActionSpec(cont_dim=2, n_discrete=1)   # [heading_rad, speed_frac | fish]
PATROL_ACTIONS = ActionSpec(cont_dim=2, n_discrete=0)    # [heading_rad, speed_frac]


class ActionDist(NamedTuple):
    mean: jax.Array       # (..., cont_dim)
    log_std: jax.Array    # (..., cont_dim)
    logits: jax.Array     # (..., n_discrete); may be zero-width


def sample_action(dist: ActionDist, key):
    k_c, k_d = jax.random.split(key)
    cont = dist.mean + jnp.exp(dist.log_std) * jax.random.normal(k_c, dist.mean.shape)
    parts = [cont]
    if dist.logits.shape[-1] > 0:
        disc = jax.random.bernoulli(k_d, jax.nn.sigmoid(dist.logits))
        parts.append(disc.astype(cont.dtype))
    return jnp.concatenate(parts, axis=-1)


def mode_action(dist: ActionDist):
    parts = [dist.mean]
    if dist.logits.shape[-1] > 0:
        parts.append((dist.logits > 0).astype(dist.mean.dtype))
    return jnp.concatenate(parts, axis=-1)


def log_prob(dist: ActionDist, action):
    c = dist.mean.shape[-1]
    a_c = action[..., :c]
    z = (a_c - dist.mean) / jnp.exp(dist.log_std)
    lp = (-0.5 * z ** 2 - dist.log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    if dist.logits.shape[-1] > 0:
        a_d = action[..., c:]
        lp = lp + (a_d * jax.nn.log_sigmoid(dist.logits)
                   + (1 - a_d) * jax.nn.log_sigmoid(-dist.logits)).sum(-1)
    return lp


def entropy(dist: ActionDist):
    h = (0.5 * jnp.log(2 * jnp.pi * jnp.e) + dist.log_std).sum(-1)
    if dist.logits.shape[-1] > 0:
        p = jax.nn.sigmoid(dist.logits)
        h = h + (-(p * jax.nn.log_sigmoid(dist.logits)
                   + (1 - p) * jax.nn.log_sigmoid(-dist.logits))).sum(-1)
    return h


# ----------------------------------------------------------------------------
# Networks
#
# Network contract (any flax module that satisfies this works):
#   net.apply(params, obs, method="actor")        -> ActionDist
#   net.apply(params, critic_obs, method="value") -> value, shape (...,)
# Inputs may have any leading batch dims, e.g. (num_envs, num_agents, dim).
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    features: Sequence[int] = (256, 256)
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for f in self.features:
            x = nn.Dense(f, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
            x = self.activation(x)
        return x


class ActorCritic(nn.Module):
    actor_backbone: nn.Module
    critic_backbone: nn.Module
    action_spec: ActionSpec

    def setup(self):
        spec = self.action_spec
        self.mean_head = nn.Dense(spec.cont_dim, kernel_init=nn.initializers.orthogonal(0.01))
        self.log_std = self.param("log_std", nn.initializers.zeros, (spec.cont_dim,))
        self.logit_head = (nn.Dense(spec.n_discrete, kernel_init=nn.initializers.orthogonal(0.01))
                           if spec.n_discrete > 0 else None)
        self.value_head = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

    def actor(self, obs) -> ActionDist:
        h = self.actor_backbone(obs)
        mean = self.mean_head(h)
        if self.logit_head is not None:
            logits = self.logit_head(h)
        else:
            logits = jnp.zeros(h.shape[:-1] + (0,))
        log_std = jnp.broadcast_to(self.log_std, mean.shape)
        return ActionDist(mean, log_std, logits)

    def value(self, critic_obs):
        return self.value_head(self.critic_backbone(critic_obs))[..., 0]

    def __call__(self, obs, critic_obs):
        return self.actor(obs), self.value(critic_obs)


NetworkFactory = Callable[[ActionSpec], nn.Module]


def make_mlp_actor_critic(hidden=(256, 256), activation=nn.tanh) -> NetworkFactory:
    def factory(action_spec: ActionSpec) -> nn.Module:
        return ActorCritic(
            actor_backbone=MLP(hidden, activation),
            critic_backbone=MLP(hidden, activation),
            action_spec=action_spec,
        )
    return factory


class TeamDims(NamedTuple):
    obs_dim: int
    critic_dim: int
    action_spec: ActionSpec


def build_policies(key, dims: dict, factory: NetworkFactory):
    """Creates one network and one parameter set per team (shared across that team's agents)."""
    nets, params = {}, {}
    for name, d in dims.items():
        key, sub = jax.random.split(key)
        net = factory(d.action_spec)
        params[name] = net.init(sub, jnp.zeros((d.obs_dim,)), jnp.zeros((d.critic_dim,)))
        nets[name] = net
    return nets, params