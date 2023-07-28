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

def reverse(sde:SDE ,score_fn: Callable, probaility_flow=False):

    N = sde.N
    T = sde.T
    sde_fn = sde.sde
    discretize_fn = sde.discretize

    @jax.vmap
    def batch_mul(x, y):
        return x*y

    # Build the class for reverse-time SDE.
    class RSDE(SDE):
        def __init__(self):
            self.N = N
            self.probability_flow = probaility_flow

        @property
        def T(self):
            return T

        def sde(self, x, t):
            """Create the drift and diffusion functions for the reverse SDE/ODE."""
            drift, diffusion = sde_fn(x, t)
            score = score_fn(x, t)
            drift = drift - batch_mul(diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
            # Set the diffusion function to zero for ODEs.
            diffusion = jnp.zeros_like(diffusion) if self.probability_flow else diffusion
            return drift, diffusion

        def discretize(self, x, t):
            """Create discretized iteration rules for the reverse diffusion sampler."""
            f, G = discretize_fn(x, t)
            rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
            rev_G = jnp.zeros_like(G) if self.probability_flow else G
            return rev_f, rev_G

    return RSDE()