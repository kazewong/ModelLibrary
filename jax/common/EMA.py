import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union

class EMA(eqx.Module):
    """
    Exponential Moving Average
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay

    def __call__(self, x: Array, y: Array) -> Array:
        """
        Update the moving average of x with y.
        """
        return self.decay * x + (1 - self.decay) * y