import equinox as eqx
import jax
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from jaxtyping import Int, Array, PRNGKeyArray
from typing import Callable, Tuple
from functools import partial


def naive_upsample(
    x: Int[Array, "dim ..."], factor: int = 2
) -> Int[Array, "factor*dim ..."]:
    current_shape = x.shape
    n_dim = len(current_shape) - 1
    x = x.reshape(
        current_shape[0],
        *tree_flatten([[current_shape[i + 1], 1] for i in range(n_dim)])[0]
    )
    x = jnp.tile(x, [1, *[1 for _ in range(n_dim)], *[factor for _ in range(n_dim)]])
    return x.reshape(
        current_shape[0], *[current_shape[i + 1] * factor for i in range(n_dim)]
    )


def naive_downsample(
    x: Int[Array, "dim ..."], factor: int = 2
) -> Int[Array, "dim/factor ..."]:
    current_shape = x.shape
    n_dim = len(current_shape) - 1
    new_shape = [[current_shape[i + 1] // factor, factor] for i in range(n_dim)]
    x = x.reshape(current_shape[0], *tree_flatten(new_shape)[0])
    x = jnp.mean(x, axis=tuple(range(2, 2 * n_dim + 1, 2)))
    return x


def fir_upsample(
    x: Int[Array, "dim ..."], factor: int = 2, kernel_size: int = 3, gain: float = 1.0
) -> Int[Array, "factor*dim ..."]:
    pass


def fir_downsample(
    x: Int[Array, "dim ..."], factor: int = 2, kernel_size: int = 3, gain: float = 1.0
) -> Int[Array, "dim/factor ..."]:
    pass


class UpDownSampling(eqx.Module):
    # TODO: implement conv upsampling and downsampling
    num_dim: int
    factor: int
    filter: Callable
    mode: str = "naive"  # 'naive', 'fir', 'conv'

    def __init__(
        self,
        num_dim: int,
        up: bool,
        factor: int = 2,
        mode: str = "naive",
        fir_kernel_size=3,
        fir_gain=1.0,
    ):
        self.num_dim = num_dim
        self.mode = mode
        self.factor = factor
        if self.mode == "naive":
            if up:
                self.filter = partial(naive_upsample, factor=factor)
            else:
                self.filter = partial(naive_downsample, factor=factor)
        elif self.mode == "fir":
            if up:
                self.filter = partial(
                    fir_upsample,
                    factor=factor,
                    kernel_size=fir_kernel_size,
                    gain=fir_gain,
                )
            else:
                self.filter = partial(
                    fir_downsample,
                    factor=factor,
                    kernel_size=fir_kernel_size,
                    gain=fir_gain,
                )

    def __call__(self, x: Array) -> Array:
        return self.filter(x)
