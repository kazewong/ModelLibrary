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
    max_channels: int = 512
    up_down_factor: int = 2
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
    UpBlocks: list[ResnetBlock]
    BottleNeck: list[ResnetBlock]
    FinalResBlock: ResnetBlock
    FinalGroupNorm: eqx.nn.Sequential

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
                        num_in_channels=min(
                            config.base_channels * 2**i_level, config.max_channels
                        ),
                        num_out_channels=min(
                            config.base_channels * 2**i_level, config.max_channels
                        ),
                    )
                )
            if i_level != config.n_resolution - 1:
                key, subkey = jax.random.split(key)
                self.DownBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=min(
                            config.base_channels * 2**i_level, config.max_channels
                        ),
                        num_out_channels=min(
                            config.base_channels * 2 ** (i_level + 1),
                            config.max_channels,
                        ),
                        sampling="down",
                    )
                )
                self.DownBlocks.append(
                    UpDownSampling(
                        num_dim=config.num_dim,
                        up=False,
                        factor=config.up_down_factor,
                        mode=config.sampling_method,
                        fir_kernel_size=config.fir_kernel_size,
                    )
                )

        for i_block in range(config.n_resnet_blocks):
            key, subkey = jax.random.split(key)
            self.BottleNeck.append(
                ResBlock(
                    key=subkey,
                    num_in_channels=min(
                        config.base_channels * 2 ** (config.n_resolution - 1),
                        config.max_channels,
                    ),
                    num_out_channels=min(
                        config.base_channels * 2 ** (config.n_resolution - 1),
                        config.max_channels,
                    ),
                )
            )

        for i_level in range(config.n_resolution):
            for i_block in range(config.n_resnet_blocks):
                key, subkey = jax.random.split(key)
                self.UpBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=min(
                            config.base_channels * 2**i_level, config.max_channels
                        ),
                        num_out_channels=min(
                            config.base_channels * 2**i_level, config.max_channels
                        ),
                    )
                )
            if i_level != config.n_resolution - 1:
                key, subkey = jax.random.split(key)
                self.UpBlocks.append(
                    ResBlock(
                        key=subkey,
                        num_in_channels=min(
                            config.base_channels * 2 ** (i_level + 1),
                            config.max_channels,
                        ),
                        num_out_channels=min(
                            config.base_channels * 2 ** (i_level), config.max_channels
                        ),
                        sampling="up",
                    )
                )
                self.UpBlocks.append(
                    UpDownSampling(
                        num_dim=config.num_dim,
                        up=True,
                        factor=config.up_down_factor,
                        mode=config.sampling_method,
                        fir_kernel_size=config.fir_kernel_size,
                    )
                )

        self.UpBlocks = list(reversed(self.UpBlocks))

        key, subkey = jax.random.split(key)
        self.FinalResBlock = ResBlock(
                    key=subkey,
                    num_in_channels=config.base_channels * 2,
                    num_out_channels=config.base_channels,
                )
        self.FinalGroupNorm = eqx.nn.Sequential(
            [
                eqx.nn.GroupNorm(
                    min(config.group_norm_size, config.output_channels),
                    config.output_channels,
                ),
                eqx.nn.Lambda(activation),
            ]
        )

    def __call__(
        self,
        x: Array,
        t: Array,
        key: PRNGKeyArray,
        train: bool = True,
    ) -> Array:
        key, subkey = jax.random.split(key)
        x_res = []
        x = self.input_conv(x)
        x_res.append(x)
        for index, block in enumerate(self.DownBlocks):
            key, subkey = jax.random.split(key)
            if type(block) == ResnetBlock:
                x = block(x, subkey, t, train=train)
                x_res.append(x)
            else:
                x = block(x)
        for block in self.BottleNeck:
            key, subkey = jax.random.split(key)
            x = block(x, subkey, t, train=train)
        for block in self.UpBlocks:
            key, subkey = jax.random.split(key)
            if type(block) == ResnetBlock:
                x = block(x + x_res.pop(), subkey, t, train=train)
            else:
                x = block(x)

        key, subkey = jax.random.split(key)
        x = jnp.concatenate([x, x_res.pop()], axis=0)
        x = self.FinalResBlock(x, subkey, t, train=train)

        x = self.output_conv(self.FinalGroupNorm(x))
        return x
