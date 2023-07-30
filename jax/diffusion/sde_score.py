import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from sde import SDE
from abc import ABC

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
    predictor: Callable
    corrector: Callable

    @property
    def n_dim(self) -> int:
        return self.autoencoder.n_dim

    def __init__(self,
                autoencoder: eqx.Module,
                time_feature: GaussianFourierFeatures,
                time_embed: eqx.nn.Linear,
                weight_function: Callable,
                sde: SDE,
                predictor: Callable = None,
                corrector: Callable = None
                ):
        self.autoencoder = autoencoder
        self.time_feature = time_feature
        self.time_embed = time_embed
        self.weight_function = weight_function
        self.sde = sde
        if predictor is None:
            predictor = EulerMaruyamaPredictor(self.sde, self.score, False)
        else:
            self.predictor = predictor
        if corrector is None:
            corrector = EulerMaruyamaCorrector(self.sde, self.score, False)
        self.corrector = corrector

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

    def sample(self,
                data_shape: tuple[int],
                key: PRNGKeyArray,
                n_steps:int,
                eps: float = 1e-3,
                ) -> tuple(Array, Array, PRNGKeyArray):
        key, subkey = jax.random.split(key)
        x_init = self.sde.sample_prior(subkey, data_shape)
        time_steps = jnp.linspace(self.sde.T, eps, n_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_mean = x_init

        for time_step in time_steps:
            key, subkey = jax.random.split(key)
            x, x_mean = self.predictor(self.sde, x, time_step, subkey, self.score, step_size)
            key, subkey = jax.random.split(key)
            x, x_mean = self.corrector(self.sde, x, time_step, subkey, self.score, step_size)
        return x, x_mean, key

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

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self,
                sde: SDE,
                score: Callable,
                probability_flow: bool = False):
    super().__init__()
    self.sde = sde
    self.score = score
    self.probability_flow = probability_flow

  @abc.abstractmethod
  def update_fn(self,
                key: PRNGKeyArray,
                x: Array,
                t: float,
                step_size: float) -> tuple[Array, Array]:
    """One update of the predictor.
    """
    pass

class EulerMaruyamaPredictor(Predictor):

    def __call__(self,  key: PRNGKeyArray, x: Array, time: float,step_size: float) -> Array:
        drift, diffusion = self.sde.reverse_sde(x, time.reshape(1), self.score)
        x_mean = x - drift * step_size
        key, subkey = jax.random.split(key)
        x = x_mean + diffusion* jnp.sqrt(step_size) * jax.random.normal(subkey, x.shape)      
        return x, x_mean

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self,
                sde: SDE,
                score: Callable,
                snr: float,
                n_steps: int):
    super().__init__()
    self.sde = sde
    self.score = score
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def __call__(self,
                key: PRNGKeyArray,
                x: Array,
                t: float,
                step_size: float) -> tuple[Array, Array]:
    """One update of the corrector.
    """
    pass

class LangevinCorrector(Corrector):


def langevin_corrector(sde: SDE, x: Array, time: float, key: PRNGKeyArray, score: Callable, step_size: float) -> Array:
    def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      grad_norm = jnp.linalg.norm(
        grad.reshape((grad.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean

def none_corrector(sde: SDE, x: Array, time: float, key: PRNGKeyArray, score: Callable, step_size: float) -> Array:
    return x, x