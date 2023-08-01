import equinox as eqx
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from abc import abstractmethod

class EmbedBase(eqx.Module):

    def __call__(self,
                x: Array,) -> Array:
        return self.embed(x)

    @abstractmethod
    def embed(self,
            x: Array,) -> Array:
        raise NotImplementedError


class ClassEmbed(EmbedBase):

    """
    Normal embedding based on a bag of vocabulary.
    TODO: This is not completed and cannot be used yet.
    """
    
    _num_embeddings: int
    _embed_dim: int
    padding_idx: int
    learned: bool = False

    def __init__(self,
                 num_embeddings: int,
                 embed_dim: int,
                 padding_idx: int = 0,
                 learned: bool = False):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.learned = learned

    def embed(self, x: Array) -> Array:
        return super().embed(x)
class PatchEmbed(EmbedBase):

    """
    Patch Embedding module used in Vision Transformer.
    """

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
                 key: PRNGKeyArray,
                 patch_size: int,
                 img_size: int,
                 in_channels: int,
                 embed_dim: int):
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
                                    key=key)

    def __call__(self, x: Array, flatten_channels: bool = True) -> Array:
        return self.embed(x, flatten_channels)

    def embed(self, x: Array, flatten_channels: bool = True) -> Array:
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
    

def test_patch_embed():
    # Create a PatchEmbed instance
    patch_embed = PatchEmbed(key=jax.random.PRNGKey(0),patch_size=16, img_size=224, in_channels=3, embed_dim=64)

    # Create a random input image tensor
    input_shape = (3, 224, 224)
    x = jnp.ones(input_shape)

    # Test the image_to_patch method
    patches = patch_embed.embed(x)
    assert patches.shape == (64, 196)

    # Test the conv layer output shape
    conv_output = patch_embed.conv(x)
    assert conv_output.shape == (64, 14, 14)