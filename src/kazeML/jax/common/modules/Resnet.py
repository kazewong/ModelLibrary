import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Tuple
from kazeML.jax.common.modules.Updown_sampling import UpDownSampling


class ResnetBlock(eqx.Module):
    act: eqx.nn.Lambda
    conv_in_block: eqx.nn.Conv
    conv_out_block: eqx.nn.Conv
    conv_residual_block: eqx.nn.Conv
    dropout: eqx.nn.Dropout
    group_norm_in: eqx.nn.GroupNorm
    group_norm_out: eqx.nn.GroupNorm
    conditional: eqx.nn.Linear | None
    skip_rescale: bool
    sampling: Callable

    def __init__(
        self,
        key: PRNGKeyArray,
        num_dim: int,
        num_in_channels: int,
        num_out_channels: int,
        conditional_size: int = 0,
        dropout: float = 0.1,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 1,
        group_norm_size: int = 32,
        activation: Callable = jax.nn.swish,
        skip_rescale: bool = True,
        sampling: str = "same",  # 'same', 'up', 'down'
        sampling_method: str = "naive",  # 'naive', 'fir', 'conv'
        fir: bool = False,
        fir_kernel_size: int = 3,
    ):
        self._n_dim = num_dim
        subkey = jax.random.split(key, 4)
        self.conv_in_block = eqx.nn.Conv(
            num_dim,
            num_in_channels,
            num_out_channels,
            kernel_size=kernel_size,
            key=subkey[0],
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.conv_out_block = eqx.nn.Conv(
            num_dim,
            num_in_channels,
            num_out_channels,
            kernel_size=kernel_size,
            key=subkey[1],
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.conv_residual_block = eqx.nn.Conv(
            num_dim,
            num_in_channels,
            num_out_channels,
            kernel_size=1,
            key=subkey[2],
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

        if conditional_size > 0:
            self.conditional = eqx.nn.Linear(
                conditional_size, num_out_channels, key=subkey[3]
            )
        else:
            self.conditional = None

        self.dropout = eqx.nn.Dropout(dropout)
        self.group_norm_in = eqx.nn.GroupNorm(
            min(group_norm_size, num_out_channels), num_out_channels
        )
        self.group_norm_out = eqx.nn.GroupNorm(
            min(group_norm_size, num_in_channels), num_in_channels
        )
        self.act = eqx.nn.Lambda(activation)
        self.skip_rescale = skip_rescale
        if sampling == "same":
            self.sampling = lambda x: x
        elif sampling == "up":
            self.sampling = UpDownSampling(
                num_dim=num_dim,
                up=True,
                factor=2,
                mode=sampling_method,
                fir_kernel_size=fir_kernel_size,
            )
        elif sampling == "down":
            self.sampling = UpDownSampling(
                num_dim=num_dim,
                up=False,
                factor=2,
                mode=sampling_method,
                fir_kernel_size=fir_kernel_size,
            )

    def __call__(
        self,
        x: Array,
        key: PRNGKeyArray,
        condition: Array | None = None,
        train: bool = True,
    ) -> Array:
        h = self.act(self.group_norm_in(x))

        h = self.sampling(h)
        x = self.sampling(x)

        h = self.conv_in_block(h)
        if self.conditional is not None and condition is not None:
            h += jnp.expand_dims(
                self.conditional(self.act(condition)),
                axis=tuple(range(1, self._n_dim + 1)),
            )
        h = self.act(self.group_norm_out(h))
        key, subkey = jax.random.split(key)
        h = self.dropout(h, key=subkey, inference=not train)
        h = self.conv_out_block(h)
        x = self.conv_residual_block(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / jnp.sqrt(2)
