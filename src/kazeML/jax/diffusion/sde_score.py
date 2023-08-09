import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from kazeML.jax.common.Unet import Unet
from kazeML.jax.diffusion.sde import SDE
from abc import ABC, abstractmethod
from tqdm import tqdm
from tap import Tap

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


class Predictor(ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self,
                sde: SDE,
                score: Callable,
                probability_flow: bool = False):
    super().__init__()
    self.sde = sde
    self.score = score
    self.probability_flow = probability_flow

  @abstractmethod
  def __call__(self,
                key: PRNGKeyArray,
                x: Array,
                t: float,
                step_size: float) -> tuple[Array, Array]:
    """One update of the predictor.
    """
    pass

class EulerMaruyamaPredictor(Predictor):

    def __call__(self,  key: PRNGKeyArray, x: Array, time: Array, step_size: float) -> tuple[Array, Array]:
        drift, diffusion = self.sde.reverse_sde(x, time.reshape(1), self.score)
        x_mean = x - drift * step_size
        key, subkey = jax.random.split(key)
        x = x_mean + diffusion* jnp.sqrt(step_size) * jax.random.normal(subkey, x.shape)      
        return x, x_mean

class ReverseDiffusionPredictor(Predictor):

    def __call__(self, key: PRNGKeyArray, x: Array, time: Array, step_size: float) -> tuple[Array, Array]:
        drift, diffusion = self.sde.reverse_discretize(x, time.reshape(1), self.score)
        x_mean = x - drift
        key, subkey = jax.random.split(key)
        x = x_mean + diffusion * jax.random.normal(subkey, x.shape)      
        return x, x_mean

class Corrector(ABC):
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

  @abstractmethod
  def __call__(self,
                key: PRNGKeyArray,
                x: Array,
                t: float,
                step_size: float) -> tuple[Array, Array]:
    """One update of the corrector.
    """
    pass

class LangevinCorrector(Corrector):

    def __call__(self, key: PRNGKeyArray, x: Array, t: Array, step_size: float) -> tuple[Array, Array]:
        x_mean = x
        for i in range(self.n_steps):
            grad = self.score(x, t.reshape(1))
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x.shape)
            grad_norm = jnp.linalg.norm(grad)
            noise_norm = jnp.linalg.norm(noise)
            step_size = (self.snr * noise_norm / grad_norm)** 2 * 2 # * alpha
            x_mean = x_mean + step_size * grad
            x = x_mean + jnp.sqrt(step_size*2) * noise
        return x, x_mean

   
class NoneCorrector(Corrector):
   
   def __call__(self, key: PRNGKeyArray, x: Array, t: float, step_size: float) -> tuple[Array, Array]:
       return x, x

class ScoreBasedSDE(eqx.Module):

    autoencoder: Unet
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
                autoencoder: Unet,
                time_feature: GaussianFourierFeatures,
                time_embed: eqx.nn.Linear,
                weight_function: Callable,
                sde: SDE,
                predictor: Predictor | None = None,
                corrector: Corrector | None = None
                ):
        self.autoencoder = autoencoder
        self.time_feature = time_feature
        self.time_embed = time_embed
        self.weight_function = weight_function
        self.sde = sde
        if predictor is None:
            self.predictor = EulerMaruyamaPredictor(self.sde, self.score, False)
        else:
            self.predictor = predictor
        if corrector is None:
            corrector = NoneCorrector(self.sde, self.score, 0, 1)
        corrector.score = self.score
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
        loss = self.weight_function(random_t)* jnp.sum((score* std+z) ** 2)
        return loss

    def score(self, x: Array, t: Array) -> Array:
        mean, std = self.sde.marginal_prob(x, t)
        feature = self.time_embed(self.time_feature(x=t))
        score = self.autoencoder(x, feature)/std
        return score

    def sample(self,
                key: PRNGKeyArray,
                data_shape: tuple[int],
                n_steps:int,
                eps: float = 1e-5,
                ) -> tuple[Array, Array, PRNGKeyArray]:
        self.predictor.score = self.score
        self.corrector.score = self.score
        key, subkey = jax.random.split(key)
        x_init = self.sde.sample_prior(subkey, data_shape)
        time_steps = jnp.linspace(self.sde.T, eps, n_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        x_mean = x_init

        for time_step in tqdm(time_steps):
            key, subkey = jax.random.split(key)
            x, x_mean = self.predictor(subkey, x, time_step, step_size)
            key, subkey = jax.random.split(key)
            x, x_mean = self.corrector(subkey, x, time_step, step_size)
        return key, x, x_mean

    def inpaint(self,
                key: PRNGKeyArray,
                data: Array,
                mask: Array,
                n_steps:int,
                eps: float = 1e-3,):
        self.predictor.score = self.score
        self.corrector.score = self.score
        key, subkey = jax.random.split(key)
        x_init = self.sde.sample_prior(subkey, data.shape) * (1. - mask)
        time_steps = jnp.linspace(self.sde.T, eps, n_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        x_mean = x_init

        for time_step in tqdm(time_steps):
            key, subkey = jax.random.split(key)
            x, x_mean = self.predictor(subkey, x, time_step, step_size)
            
            key, subkey = jax.random.split(key)
            masked_data_mean, std = self.sde.marginal_prob(data, time_step)
            masked_data = masked_data_mean + std * jax.random.normal(subkey, data.shape)
            x = x * (1. - mask) + masked_data * mask
            x_mean = x_mean * (1. - mask) + masked_data_mean * mask

            key, subkey = jax.random.split(key)
            x, x_mean = self.corrector(subkey, x, time_step, step_size)

            key, subkey = jax.random.split(key)
            mask_data = masked_data_mean + std * jax.random.normal(subkey, data.shape) # Not sure if resampling is necessary
            x = x * (1. - mask) + mask_data * mask
            x_mean = x_mean * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    def conditional_sample(self,
                            key: PRNGKeyArray,
                            conditional_function: Callable,
                            condtional_data: Array,
                            data_shape: tuple[int],
                            n_steps:int,
                            eps: float = 1e-3,):
        """
        Conditional sampling

        Args:
            key (PRNGKeyArray): JAX random key
            conditional_function (Callable): Function that takes in x, t, y and returns a score
            condtional_data (Array): Conditional data y
            data_shape (tuple[int]): Shape of the data
            n_steps (int): Number of steps
            eps (float, optional): Epsilon. Defaults to 1e-3.
        """

        conditional_function = eqx.Partial(conditional_function, y=condtional_data)
        self.predictor.score = lambda x, t: self.score(x,t) + conditional_function(x, t)
        self.corrector.score = lambda x, t: self.score(x,t) + conditional_function(x, t)
        key, subkey = jax.random.split(key)
        x_init = self.sde.sample_prior(subkey, data_shape)
        time_steps = jnp.linspace(self.sde.T, eps, n_steps)
        step_size = time_steps[0] - time_steps[1]
        x = x_init
        x_mean = x_init

        for time_step in tqdm(time_steps):
            key, subkey = jax.random.split(key)
            x, x_mean = self.predictor(subkey, x, time_step, step_size)
            key, subkey = jax.random.split(key)
            x, x_mean = self.corrector(subkey, x, time_step, step_size)
        return key, x, x_mean
    
    def evaluate_likelihood(self):
        raise NotImplementedError

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str):
        return eqx.tree_deserialise_leaves(path+".eqx", self)
