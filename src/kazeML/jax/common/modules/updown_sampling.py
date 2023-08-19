import equinox as eqx
import jax
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from jaxtyping import Int, Array, PRNGKeyArray
from typing import Callable, Tuple

def naive_upsample(x: Int[Array, "dim ..."],
                   factor: int = 2) -> Int[Array, "factor*dim ..."]:
    current_shape = x.shape
    n_dim = len(current_shape) - 1
    x = x.reshape(current_shape[0], *tree_flatten([[current_shape[i+1], 1] for i in range(n_dim)])[0])
    x = jnp.tile(x, [1, *[1 for _ in range(n_dim)], *[factor for _ in range(n_dim)]])
    return x.reshape(current_shape[0], *[current_shape[i+1]*factor for i in range(n_dim)])


class UpDownSampling(eqx.Module):

    num_dim: int
    factor: int
    filter: Callable
    mode: str = 'naive' # 'naive', 'fir', 'conv'

    def __init__(self,
                 num_dim: int,
                 up: bool,
                 factor : int = 2,
                 mode: str = 'naive',):
        self.num_dim = num_dim
        self.mode = mode
        self.factor = factor
        if self.mode == "naive":
            

    def __call__(self, x: Array) -> Array:
        return self.filter(x, self.factor)


class FIRFilter(eqx.Module):

    def __init__(self,
                 num_dim: int,
                 upsampling_factor: int,
                 downsampling_factor: int,
                 padding_size: int,):
        pass

    def __call__(self, x: Array) -> Array: