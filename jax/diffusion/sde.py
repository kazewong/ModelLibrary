import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union
import tqdm
from abc import abstractmethod

class SDE(eqx.Module):
    """
    Stochastic Differential Equation (SDE) class.
    Adopted from https://github.com/yang-song/score_sde
    """

    N: int # Number of time steps

    def __init__(self, N: int):
        super().__init__()
        self.N = N

    @property
    @abstractmethod
    def T(self) -> float:
        return 1

    def __call__(self,
                x: Array,
                t: float) -> Array:
        return self.sde(x, t)

    @abstractmethod
    def sde(self,
            x: Array,
            t: float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def marginal_prob(self,
                      x: Array,
                      t: float) -> Array:
        pass

    @abstractmethod
    def sample_prior(self, rng: PRNGKeyArray, shape: tuple) -> Array:
        pass

    @abstractmethod
    def logp_prior(self, z: Array) -> Array:
        pass

    def discretize(self, x: Array, t: float) -> tuple[Callable, Callable]:
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G

    def reverse(self, score_fn: Callable, proability_flow=False):

        N = self.N
        T = self.T
        sde = self.sde