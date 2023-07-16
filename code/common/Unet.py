import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray
from typing import Callable

class UnetBlock(eqx.Module):

    conv_block: eqx.nn.Conv
    act: Callable = jax.nn.swish

    def __init__(self,
                num_dim: int,
                num_in_channels: int,
                num_out_channels: int,
                key: PRNGKeyArray,
                **kwargs,
                ):
        pass

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