import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable


class UnetBlock(eqx.Module):

    _n_dim: int
    conv_block: eqx.nn.Conv
    conv_transpose_block: eqx.nn.ConvTranspose
    group_norm_in: eqx.nn.GroupNorm
    group_norm_out: eqx.nn.GroupNorm
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
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
        subkey = jax.random.split(key, 5)
        self.conv_block = eqx.nn.Conv(
            num_dim, num_in_channels, num_out_channels, kernel_size=kernel_size, key=subkey[0], use_bias=False, padding=1, **kwargs)
        self.conv_transpose_block = eqx.nn.ConvTranspose(
            num_dim, num_out_channels, num_in_channels, kernel_size=kernel_size, key=subkey[1], use_bias=False, padding=1,  **kwargs)
        self.group_norm_in = eqx.nn.GroupNorm(min(group_norm_size, num_out_channels), num_out_channels)
        self.group_norm_out = eqx.nn.GroupNorm(min(group_norm_size, num_in_channels), num_in_channels)
        self.linear_in = eqx.nn.Linear(1, num_out_channels, key=subkey[2])
        self.linear_out = eqx.nn.Linear(1, num_in_channels, key=subkey[3])
        self.act = activation

    def __call__(self, x: Array, t: Array) -> Array:
        x = self.conv_block(x)
        x += jnp.expand_dims(self.linear_in(t), tuple(range(1, self.n_dim + 1)))
        x = self.group_norm_in(x)
        x = self.act(x)
        return x
    
    def encode(self, x: Array, t: Array) -> Array:
        return self(x, t)
    
    def decode(self, x: Array, t: Array) -> Array:
        x = self.conv_transpose_block(x)
        x += jnp.expand_dims(self.linear_out(t), tuple(range(1, self.n_dim + 1)))
        x = self.group_norm_out(x)
        x = self.act(x)
        return x


class Unet(eqx.Module):

    blocks: list[UnetBlock]
    conv_out: eqx.nn.Conv

    @property
    def n_dim(self):
        return self.blocks[0].n_dim

    def __init__(self,
                 num_dim: int,
                 channels: list[int],
                 key: PRNGKeyArray,
                 **kwargs,
                 ):
        self.blocks = []
        key, subkey = jax.random.split(key)
        for i in range(len(channels) - 1):
            self.blocks.append(UnetBlock(num_dim, channels[i], channels[i + 1], key, **kwargs))
        self.conv_out = eqx.nn.Conv(num_dim, channels[1], channels[0], padding=1, kernel_size= 3, key=subkey)
        
    def __call__(self, x: Array, t: Array) -> Array:
        latent = []
        for block in self.blocks[:-1]:
            x = block.encode(x, t)
            latent.append(x)
        x = self.blocks[-1].encode(x, t)
        for block in reversed(self.blocks[1:]):
            x = block.decode(x, t)
            x = x + latent.pop()
        x = self.conv_out(x)
        return x

