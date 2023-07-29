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
                t: float) -> tuple[Array,Array]:
        return self.sde(x, t)

    def sde(self,
            x: Array,
            t: float) -> tuple[Array,Array]:
        return self.drift(x, t), self.diffusion(x, t)

    @abstractmethod
    def drift(self,
            x: Array,
            t: float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def diffusion(self,
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

    def discretize(self, x: Array, t: float) -> tuple[Array, Array]:
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G

def reverse_sde(sde:SDE ,score_fn: Callable, probaility_flow=False):

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

class VPSDE(SDE):
    pass

class subVPSDE(SDE):
    pass

class VESDE(SDE):

    def __init__(self,
                sigma_min: float = 0.01,
                sigma_max: float = 50,
                N: int = 1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(jnp.linspace(jnp.log(self.sigma_min), jnp.log(self.sigma_max), N))

    def drift(self, x: Array, t: float) -> Array:
        return jnp.zeros_like(x)

    def diffusion(self, x: Array, t: float) -> Array:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))

    def marginal_prob(self, x: Array, t: float) -> tuple[Array, float]:
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def prior_sampling(self, rng: PRNGKeyArray, shape: tuple):
        return jax.random.normal(rng, shape) * self.sigma_max

    def prior_logp(self, z: Array):
        shape = z.shape
        N = jnp.prod(shape[1:])
        logp_fn = lambda z: -N / 2. * jnp.log(2 * jnp.pi * self.sigma_max ** 2) - jnp.sum(z ** 2) / (2 * self.sigma_max ** 2)
        return jax.vmap(logp_fn)(z)

    def discretize(self, x: Array, t: Array) -> tuple[Array, Array]:
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = jnp.where(timestep == 0, jnp.zeros_like(timestep), self.discrete_sigmas[timestep - 1])
        f = jnp.zeros_like(x)
        G = jnp.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G