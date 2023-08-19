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
    layer_norm: eqx.nn.LayerNorm
    act: Callable

    @property
    def n_dim(self):
        return jax.lax.stop_gradient(self._n_dim)

    def __init__(
        self,
        num_dim: int,
        num_in_channels: int,
        num_out_channels: int,
        embedding_dim: int,
        key: PRNGKeyArray,
        activation: Callable = jax.nn.swish,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_norm_size: int = 4,
        layer_norm: bool = False,
        use_bias: bool = False,
        **kwargs,
    ):
        self._n_dim = num_dim
        subkey = jax.random.split(key, 5)
        self.conv_block = eqx.nn.Conv(
            num_dim,
            num_in_channels,
            num_out_channels,
            kernel_size=kernel_size,
            key=subkey[0],
            stride=stride,
            dilation=dilation,
            use_bias=use_bias,
            padding=1,
            **kwargs,
        )
        self.conv_transpose_block = eqx.nn.ConvTranspose(
            num_dim,
            num_out_channels,
            num_in_channels,
            kernel_size=kernel_size,
            key=subkey[1],
            stride=stride,
            dilation=dilation,
            use_bias=use_bias,
            padding=1,
            **kwargs,
        )
        self.group_norm_in = eqx.nn.GroupNorm(
            min(group_norm_size, num_out_channels), num_out_channels
        )
        self.group_norm_out = eqx.nn.GroupNorm(
            min(group_norm_size, num_in_channels), num_in_channels
        )
        self.layer_norm = (
            eqx.nn.LayerNorm(None, use_bias=False, use_weight=False)
            if layer_norm
            else None
        )
        self.linear_in = eqx.nn.Linear(embedding_dim, num_out_channels, key=subkey[2])
        self.linear_out = eqx.nn.Linear(embedding_dim, num_in_channels, key=subkey[3])
        self.act = activation

    def __call__(self, x: Array, t: Array) -> Array:
        x = self.encode(x, t)
        x = self.decode(x, t)
        return x

    def encode(self, x: Array, t: Array) -> Array:
        x = self.conv_block(x)
        x += jnp.expand_dims(self.linear_in(t), tuple(range(1, self.n_dim + 1)))
        x = self.group_norm_in(x)
        x = self.act(x)
        return x

    def decode(self, x: Array, t: Array) -> Array:
        x = self.conv_transpose_block(x)
        x += jnp.expand_dims(self.linear_out(t), tuple(range(1, self.n_dim + 1)))
        x = self.group_norm_out(x)
        x = self.act(x)
        x = self.layer_norm(x) if self.layer_norm is not None else x
        return x


class Unet(eqx.Module):
    blocks: list[UnetBlock]
    conv_out: eqx.nn.Conv

    @property
    def n_dim(self):
        return self.blocks[0].n_dim

    @property
    def embedding_dim(self):
        return self.blocks[0].linear_in.in_features

    def __init__(
        self,
        num_dim: int,
        channels: list[int],
        embedding_dim: int,
        key: PRNGKeyArray,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        dilation: int | list[int] = 1,
        layer_norm: bool = False,
        **kwargs,
    ):
        self.blocks = []

        for i in range(len(channels) - 1):
            key, subkey = jax.random.split(key)
            if isinstance(stride, list):
                stride_local = stride[i]
            else:
                stride_local = stride
            if isinstance(dilation, list):
                dilation_local = dilation[i]
            else:
                dilation_local = dilation
            if isinstance(kernel_size, list):
                kernel_size_local = kernel_size[i]
            else:
                kernel_size_local = kernel_size
            self.blocks.append(
                UnetBlock(
                    num_dim,
                    channels[i],
                    channels[i + 1],
                    embedding_dim=embedding_dim,
                    key=subkey,
                    stride=stride_local,
                    dilation=dilation_local,
                    layer_norm=layer_norm,
                )
            )
        key, subkey = jax.random.split(key)
        self.conv_out = eqx.nn.Conv(
            num_dim, channels[0], channels[0], padding=0, kernel_size=1, key=subkey
        )

    def __call__(self, x: Array, t: Array) -> Array:
        x_shape = x.shape
        latent = []
        for block in self.blocks[:-1]:
            x = block.encode(x, t)
            latent.append(x)
        x = self.blocks[-1].encode(x, t)
        x = self.blocks[-1].decode(x, t)
        for block in reversed(self.blocks[1:-1]):
            x = x + latent.pop()
            x = block.decode(x, t)
        x = self.blocks[0].decode(x, t)
        x = self.conv_out(x)
        return x
