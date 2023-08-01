# Vision Model of Data2Vec
# Inspired by Fairseq's Data2Vec vision Model

import equinox as eqx
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union, Optional

from kazeML.jax.common.modules.PatchEmbed import PatchEmbed

@dataclass
class Data2VecVisionConfig:

    seed: int = 0

    layer_scale_init_value: float = field(
        default=1e-4, metadata={"help": "rescale layer outputs, 0 to disable"}
    )
    num_mask_patches: int = field(
        default=75,
        metadata={"help": "number of the visual tokens/patches need be masked"},
    )
    min_mask_patches_per_block: int = 16
    max_mask_patches_per_block: int = 196
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    shared_rel_pos_bias: bool = True

    drop_path: float = 0.1
    attention_dropout: float = 0.0
    embed_dim: int = 768




class Data2VecVision(eqx.Module):

    patch_embed: PatchEmbed
    class_embedding: Array
    mask_embedding: Array
    encoder: eqx.Module
    final_projection: eqx.nn.Linear

    def __init__(self, cfg: Data2VecVisionConfig):
        super().__init__()

        key = jax.random.PRNGKey(cfg.seed)
        key, subkey = jax.random.split(key)
        self.patch_embed = PatchEmbed(key=subkey,
                                      patch_size=cfg.patch_size,
                                      img_size=cfg.image_size,
                                      in_channels=cfg.in_channels,
                                      embed_dim=cfg.embed_dim)
        
        key, subkey = jax.random.split(key)
        self.class_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        key, subkey = jax.random.split(key)
        self.mask_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        

    def make_ema_teacher(self):
        pass

    def forward(self,
                img: Array,
                mask: bool = True) -> Array:
        x = self.patch_embed(img)