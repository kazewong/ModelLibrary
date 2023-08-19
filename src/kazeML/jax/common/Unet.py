from dataclasses import dataclass
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from kazeML.jax.common.modules.Updown_sampling import UpDownSampling
from kazeML.jax.common.modules.Resnet import ResnetBlock


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


@dataclass
class UnetConfig:
    num_dim: int
    embedding_dim: int
    base_channels: int = 4
    n_resolution: int = 4
    n_resnet_blocks: int = 2
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    group_norm_size: int = 32
    dropout: float = 0.1
    padding: int = 1
    activation: Callable = jax.nn.swish
    skip_rescale: bool = True
    sampling_method: str = "naive"
    fir: bool = False
    fir_kernel_size: int = 3


class Unet(eqx.Module):
    DownBlocks: list[ResnetBlock]
    UpBlocks: list[ResnetBlock]
    BottleNeck: list[ResnetBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        config: UnetConfig,
    ):
        key = key
        self.DownBlocks = []
        self.UpBlocks = []
        self.BottleNeck = []

        ResBlock = partial(
            ResnetBlock,
            num_dim=config.num_dim,
            conditional_size=config.embedding_dim,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
            stride=config.stride,
            dilation=config.dilation,
            padding=config.padding,
            group_norm_size=config.group_norm_size,
            activation=config.activation,
            skip_rescale=config.skip_rescale,
            sampling_method=config.sampling_method,
            fir=config.fir,
            fir_kernel_size=config.fir_kernel_size,
        )

        for i_level in range(config.n_resolution):
            for i_block in range(config.n_resnet_blocks):
                key, subkey = jax.random.split(key)
                self.DownBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=config.base_channels * 2**i_level,
                        num_out_channels=config.base_channels * 2**i_level,
                    )
                )
            if i_level != config.n_resolution - 1:
                key, subkey = jax.random.split(key)
                self.DownBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=config.base_channels * 2**i_level,
                        num_out_channels=config.base_channels * 2 ** (i_level + 1),
                        sampling_method="down",
                    )
                )

        for i_block in range(config.n_resnet_blocks):
            key, subkey = jax.random.split(key)
            self.BottleNeck.append(
                ResBlock(
                    key=subkey,
                    num_in_channels=config.base_channels * 2 ** (config.n_resolution),
                    num_out_channels=config.base_channels * 2 ** (config.n_resolution),
                )
            )

        for i_level in reversed(range(config.n_resolution)):
            for i_block in range(config.n_resnet_blocks):
                key, subkey = jax.random.split(key)
                self.UpBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=config.base_channels * 2 ** (i_level + 1),
                        num_out_channels=config.base_channels * 2**i_level,
                    )
                )
            if i_level != 0:
                key, subkey = jax.random.split(key)
                self.UpBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=config.base_channels * 2**i_level,
                        num_out_channels=config.base_channels * 2 ** (i_level - 1),
                        sampling_method="up",
                    )
                )

    def __call__(
        self,
        x: Array,
        t: Array,
        key: PRNGKeyArray,
        train: bool = True,
    ) -> Array:
        for block in self.DownBlocks:
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
        for block in self.BottleNeck:
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
        for block in self.UpBlocks:
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
        return x
