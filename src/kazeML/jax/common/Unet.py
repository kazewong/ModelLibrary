from dataclasses import dataclass
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from kazeML.jax.common.modules.Updown_sampling import UpDownSampling
from kazeML.jax.common.modules.Resnet import ResnetBlock


@dataclass
class UnetConfig:
    num_dim: int
    input_channels: int
    output_channels: int
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
    skip_rescale: bool = True
    sampling_method: str = "naive"
    fir: bool = False
    fir_kernel_size: int = 3


class Unet(eqx.Module):

    # TODO: add pyramid sampling
    input_conv: eqx.nn.Conv
    output_conv: eqx.nn.Conv
    DownBlocks: list[ResnetBlock]
    PyramidDownBlocks: list[eqx.nn.Sequential]
    UpBlocks: list[ResnetBlock]
    PyramidUpBlocks: list[eqx.nn.Sequential]
    BottleNeck: list[ResnetBlock]
    n_resolution: int
    n_resnet_blocks: int

    @property
    def n_dim(self) -> int:
        return self.input_conv.num_spatial_dims

    def __init__(
        self,
        key: PRNGKeyArray,
        config: UnetConfig,
        activation: Callable = jax.nn.swish,
    ):
        key = key
        self.DownBlocks = []
        self.UpBlocks = []
        self.BottleNeck = []
        self.PyramidDownBlocks = []
        self.PyramidUpBlocks = []
        self.n_resolution = config.n_resolution
        self.n_resnet_blocks = config.n_resnet_blocks

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
            activation=eqx.nn.Lambda(activation),
            skip_rescale=config.skip_rescale,
            sampling_method=config.sampling_method,
            fir=config.fir,
            fir_kernel_size=config.fir_kernel_size,
        )

        key, subkey = jax.random.split(key)
        self.input_conv = eqx.nn.Conv(
            config.num_dim,
            config.input_channels,
            config.base_channels,
            kernel_size=1,
            key=subkey,
        )
        self.output_conv = eqx.nn.Conv(
            config.num_dim,
            config.base_channels,
            config.output_channels,
            kernel_size=1,
            key=subkey,
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
                        sampling="down",
                    )
                )
                key, subkey = jax.random.split(key)
                self.PyramidDownBlocks.append(
                    eqx.nn.Sequential([eqx.nn.Conv(
                            config.num_dim,
                            in_channels=config.base_channels * 2 ** (i_level),
                            out_channels=config.base_channels * 2 ** (i_level + 1),
                            kernel_size=1,
                            key=subkey,
                    ),
                    UpDownSampling(
                        config.num_dim,
                        up=False,
                        mode=config.sampling_method,
                        fir_kernel_size=config.fir_kernel_size,
                    )]))

        for i_block in range(config.n_resnet_blocks):
            key, subkey = jax.random.split(key)
            self.BottleNeck.append(
                ResBlock(
                    key=subkey,
                    num_in_channels=config.base_channels
                    * 2 ** (config.n_resolution - 1),
                    num_out_channels=config.base_channels
                    * 2 ** (config.n_resolution - 1),
                )
            )

        for i_level in reversed(range(config.n_resolution)):
            for i_block in range(config.n_resnet_blocks):
                key, subkey = jax.random.split(key)
                self.UpBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=config.base_channels * 2**i_level,
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
                        sampling="up",
                    )
                )
                key, subkey = jax.random.split(key)
                self.PyramidUpBlocks.append(
                    eqx.nn.Sequential([eqx.nn.Conv(
                            config.num_dim,
                            in_channels=config.base_channels * 2 ** (i_level),
                            out_channels=config.base_channels * 2 ** (i_level - 1),
                            kernel_size=1,
                            key=subkey,
                    ),
                    UpDownSampling(
                        config.num_dim,
                        up=True,
                        mode=config.sampling_method,
                        fir_kernel_size=config.fir_kernel_size,
                    )]))

    def __call__(
        self,
        x: Array,
        t: Array,
        key: PRNGKeyArray,
        train: bool = True,
    ) -> Array:
        key, subkey = jax.random.split(key)
        x = self.input_conv(x)
        pyramid = x
        x_res = []
        for index, block in enumerate(self.DownBlocks):
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
            if index % (self.n_resnet_blocks+1) == self.n_resnet_blocks:
                pyramid = self.PyramidDownBlocks[index//(self.n_resnet_blocks+1)](pyramid)
                x = x + pyramid
            x_res.append(x)
        for block in self.BottleNeck:
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
        pyramid = x
        for index, block in enumerate(self.UpBlocks):
            key, subkey = jax.random.split(key)
            x = block(x+x_res.pop(), subkey, t, train=train)
            if index % (self.n_resnet_blocks+1) == self.n_resnet_blocks:
                pyramid = self.PyramidUpBlocks[index//(self.n_resnet_blocks+1)](pyramid) + x
                x = pyramid
        x = self.output_conv(x)
        return x
