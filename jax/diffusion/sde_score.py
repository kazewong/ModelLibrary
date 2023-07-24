import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union
import tqdm

class GaussianFourierFeatures(eqx.Module):

    weight: Array

    @property
    def n_dim(self) -> int:
        return self.weight.shape[0]

    def __init__(self,
                embed_dim: int,
                key: PRNGKeyArray,
                scale: float = 30.0,
                ):
        self.weight = jax.random.normal(key, (embed_dim // 2, )) * scale

    def __call__(self, x: Array) -> Array:
        weight = jax.lax.stop_gradient(self.weight)
        x_proj = x[:, None] * weight[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)[0]


class ScordBasedSDE(eqx.Module):

    autoencoder: eqx.Module
    time_feature: eqx.Module
    time_embed: eqx.nn.Linear
    weight_function: Callable
    drift_function: Callable
    diffusion_function: Callable
    marginal_prob: Callable

    @property
    def n_dim(self) -> int:
        return self.autoencoder.n_dim

    def __init__(self,
                autoencoder: eqx.Module,
                weight_function: Callable,
                drift_function: Callable,
                diffusion_function: Callable,
                marginal_prob: Callable,
                time_feature: GaussianFourierFeatures,
                time_embed: eqx.nn.Linear,
                ):
        self.autoencoder = autoencoder
        self.weight_function = weight_function
        self.drift_function = drift_function
        self.diffusion_function = diffusion_function
        self.marginal_prob = marginal_prob
        self.time_feature = time_feature
        self.time_embed = time_embed

    def __call__(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        return self.loss(x, key, eps)
        
    def loss(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        # Loss for one data point
        key, subkey = jax.random.split(key)
        random_t = jax.random.uniform(subkey, (1,), minval=eps, maxval=1.0)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, x.shape)
        std = self.marginal_prob(random_t)
        perturbed_x = x + std * z
        score = self.score(perturbed_x, random_t)
        loss = self.weight_function(random_t)* jnp.sum((score*std+z) ** 2)
        return loss

    def score(self, x: Array, t: Array) -> Array:
        std = self.marginal_prob(t)
        time_feature = jax.nn.swish(self.time_embed(self.time_feature(x=t)))
        score = self.autoencoder(x, time_feature) / std
        return score

    def sample(self, data_shape: tuple[int], key: PRNGKeyArray, num_steps:int = 500, batch_size:int = 1, eps: float = 1e-3) -> Array:
        score_map = jax.vmap(self.score)
        key, subkey = jax.random.split(key)
        time_shape = (batch_size,)
        sample_shape = time_shape + data_shape
        init_x = jax.random.normal(subkey, sample_shape) * self.diffusion_function(1.)
        time_steps = jnp.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        mean_x = init_x
        for time_step in tqdm.tqdm(time_steps):      
            batch_time_step = jnp.ones(time_shape+(1,)) * time_step
            g = self.diffusion_function(time_step)
            mean_x = x + (g**2) * score_map(x, batch_time_step) * step_size
            key, subkey = jax.random.split(key)
            x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(subkey, x.shape)      
        # Do not include any noise in the last sampling step.
        return mean_x

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)