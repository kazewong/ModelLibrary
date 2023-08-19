import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Tuple

class FIRFilter(eqx.Module):

    def __init__(self,
                 num_dim: int,
                 upsampling_factor: int,
                 downsampling_factor: int,
                 padding_size: int,):
        pass

    def __call__(self, x: Array) -> Array: