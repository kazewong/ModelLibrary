import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

class GaussianFourierFeatures(eqx.Module):

    weight: Array

    def __init__(self,
                embed_dim: int,
                key: PRNGKeyArray,
                scale: float = 30.0,
                ):
        self.weight = jax.random.normal(key, (embed_dim // 2, )) * scale

    def __call__(self, x: Array) -> Array:
        weight = jax.lax.stop_gradient(self.weight)
        x_proj = x[:, None] * weight[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class ScordBasedSDE(eqx.Module):

    blocks: list

    def __init__(self,
                 blocks: list,
                 key: PRNGKeyArray,
                 ):
        self.blocks = blocks

    def __call__(self, x: Array, t: Array) -> Array:
        pass

    def score(self):
        pass

    def sample(self):
        pass