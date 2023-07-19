from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import diffrax

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

    autoencoder: eqx.Module
    weight_function: Callable
    drift_function: Callable
    diffusion_function: Callable

    @property
    def n_dim(self) -> int:
        return self.autoencoder.n_dim

    def __init__(self,
                autoencoder: eqx.Module,
                weight_function: Callable,
                drift_function: Callable,
                diffusion_function: Callable,
                ):
        self.autoencoder = autoencoder
        self.weight_function = weight_function
        self.drift_function = drift_function
        self.diffusion_function = diffusion_function

    def __call__(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        return self.score(x, key, eps)
        
    def loss(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        key, subkey = jax.random.split(key)
        random_t = jax.random.uniform(subkey, (1,), minval=eps, maxval=1.0)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, x.shape)
        std = jnp.sqrt(self.diffusion_function(random_t))
        perturbed_x = x + std * z
        score = self.score(perturbed_x, random_t)
        loss = self.weight_function(random_t)* jnp.mean(jnp.sum((score*std+z) ** 2, axis=range(1,self.n_dim+1)))
        return loss


    def score(self, x: Array, t: Array) -> Array:
        std = jnp.sqrt(self.diffusion_function(t))
        score = self.autoencoder(x, t) / std
        return score

    def sample(self):
        pass