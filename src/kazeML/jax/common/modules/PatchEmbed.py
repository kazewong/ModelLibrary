import equinox as eqx
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union, Optional

class PatchEmbed(eqx.Module):

    _patch_size: int
    _img_size: int
    _in_channels: int
    _embed_dim: int
    conv: eqx.nn.Conv2d

    @property
    def patch_size(self) -> int:
        return jax.lax.stop_gradient(self._patch_size)
    
    @property
    def img_size(self) -> int:
        return jax.lax.stop_gradient(self._img_size)
    
    @property
    def in_channels(self) -> int:
        return jax.lax.stop_gradient(self._in_channels)
    
    @property
    def embed_dim(self) -> int:
        return jax.lax.stop_gradient(self._embed_dim)

    def __init__(self,
                 patch_size: int,
                 img_size: int,
                 in_channels: int,
                 embed_dim: int,):
        super().__init__()
        self._patch_size = patch_size
        self._img_size = img_size
        self._in_channels = in_channels
        self._embed_dim = embed_dim

        self.conv = eqx.nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    padding=0,
                                    bias=True)


    def image_to_patch(self, x: Array, flatten_channels: bool = True) -> Array:
        """
        Args:
            x (Array): Input image of shape (Channels, Height, Width)

        Returns:
            Array: Image patches of shape (Height*Width/(Patch_Size*Patch_Size), Patch_Size*Patch_Size*Num_Channels)
        """
        x = self.conv(x)
        if flatten_channels:
            x = x.reshape(x.shape[0], -1)
        return x