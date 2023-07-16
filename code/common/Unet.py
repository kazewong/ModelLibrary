import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray
from typing import Callable


class UnetBlock(eqx.Module):

    _n_dim: int
    conv_block: eqx.nn.Conv
    conv_transpose_block: eqx.nn.ConvTranspose
    group_norm_in: eqx.nn.GroupNorm
    group_norm_out: eqx.nn.GroupNorm
    act: Callable

    @property
    def n_dim(self):
        return jax.lax.stop_gradient(self._n_dim)

    def __init__(self,
                 num_dim: int,
                 num_in_channels: int,
                 num_out_channels: int,
                 key: PRNGKeyArray,
                 activation: Callable = jax.nn.swish,
                 kernel_size: int = 3,
                 group_norm_size: int = 4,
                 **kwargs,
                 ):
        self._n_dim = num_dim
        self.conv_block = eqx.nn.Conv(
            num_dim, num_in_channels, num_out_channels, kernel_size=kernel_size, key=key, **kwargs)
        self.conv_transpose_block = eqx.nn.ConvTranspose(
            num_dim, num_out_channels, num_in_channels, kernel_size=kernel_size, key=key, **kwargs)
        self.group_norm_in = eqx.nn.GroupNorm(min(group_norm_size, num_out_channels), num_out_channels)
        self.group_norm_out = eqx.nn.GroupNorm(min(group_norm_size, num_in_channels), num_in_channels)
        self.act = activation

    def __call__(self, x: Array) -> Array:
        x = self.conv_block(x)
        x = self.group_norm_in(x)
        x = self.act(x)
        return x
    
    def encode(self, x: Array) -> Array:
        return self(x)
    
    def decode(self, x: Array) -> Array:
        x = self.conv_transpose_block(x)
        x = self.group_norm_out(x)
        x = self.act(x)
        return x


class Unet(eqx.Module):

    blocks: list[UnetBlock]

    def __init__(self,
                 num_dim: int,
                 num_in_channels: int,
                 num_out_channels: int,
                 key: PRNGKeyArray,
                 **kwargs,
                 ):
        pass
