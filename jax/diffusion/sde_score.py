import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from sde import SDE
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
    time_feature: GaussianFourierFeatures
    time_embed: eqx.nn.Linear
    weight_function: Callable
    sde: SDE

    @property
    def n_dim(self) -> int:
        return self.autoencoder.n_dim

    def __init__(self,
                autoencoder: eqx.Module,
                time_feature: GaussianFourierFeatures,
                time_embed: eqx.nn.Linear,
                weight_function: Callable,
                sde: SDE,
                ):
        self.autoencoder = autoencoder
        self.time_feature = time_feature
        self.time_embed = time_embed
        self.weight_function = weight_function
        self.sde = sde

    def __call__(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        return self.loss(x, key, eps)
        
    def loss(self, x: Array, key: PRNGKeyArray, eps: float = 1e-5) -> Array:
        # Loss for one data point
        key, subkey = jax.random.split(key)
        random_t = jax.random.uniform(subkey, (1,), minval=eps, maxval=1.0)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, x.shape)
        mean, std = self.sde.marginal_prob(x, random_t)
        perturbed_x = mean + std * z
        score = self.score(perturbed_x, random_t)
        loss = self.weight_function(random_t)* jnp.sum((score*std+z) ** 2)
        return loss

    def score(self, x: Array, t: float) -> Array:
        mean, std = self.sde.marginal_prob(x, t)
        time_feature = self.time_embed(self.time_feature(x=t))
        score = self.autoencoder(x, time_feature)/std
        return score

    def sample(self, data_shape: tuple[int], key: PRNGKeyArray, eps: float = 1e-3) -> Array:
        key, subkey = jax.random.split(key)
        x_init = self.sde.sample_prior(subkey, data_shape)
        time_steps = jnp.linspace(self.sde.T, eps, self.sde.N)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        x_mean = x_init
        for time_step in tqdm.tqdm(time_steps):      
            drift, diffusion = self.sde.reverse_sde(self.score)(x, time_step)
            x_mean = x + drift * step_size
            key, subkey = jax.random.split(key)
            x = x_mean + diffusion* jnp.sqrt(step_size) * jax.random.normal(subkey, x.shape)      
        # Do not include any noise in the last sampling step.
        return x_mean

    def inpaint(self):
        raise NotImplementedError

    def colorize(self):
        raise NotImplementedError

    def conditional_sample(self):
        raise NotImplementedError
    
    def evaluate_likelihood(self):
        raise NotImplementedError

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str):
        eqx.tree_deserialise_leaves(path+".eqx", self)