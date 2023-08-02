# Vision Model of Data2Vec
# Inspired by Fairseq's Data2Vec vision Model

import equinox as eqx
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from kazeML.jax.common.Transformer import TransformerConfig, TransformerEncoder

from kazeML.jax.common.modules.Embedding import PatchEmbedding


@dataclass
class Data2VecVisionConfig:

    transformer_decoder_config: TransformerConfig

    layer_scale_init_value: float = 1e-4
    num_mask_patches: int = 75
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

    patch_embed: PatchEmbedding
    class_embedding: Array
    mask_embedding: Array
    encoder: TransformerEncoder
    final_projection: eqx.nn.Linear

    def __init__(self, key: PRNGKeyArray,
                cfg: Data2VecVisionConfig):
        super().__init__()

        key, subkey = jax.random.split(key)
        self.patch_embed = PatchEmbedding(key=subkey,
                                      patch_size=cfg.patch_size,
                                      img_size=cfg.image_size,
                                      in_channels=cfg.in_channels,
                                      embed_dim=cfg.embed_dim)
        
        key, subkey = jax.random.split(key)
        self.class_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        key, subkey = jax.random.split(key)
        self.mask_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        key, subkey = jax.random.split(key)
        self.encoder = TransformerEncoder(subkey, cfg.transformer_decoder_config, self.patch_embed)
        
        key, subkey = jax.random.split(key)
        self.final_projection = eqx.nn.Linear(key=subkey, in_features=cfg.embed_dim, out_features=cfg.embed_dim)

    def make_ema_teacher(self):
        pass

    def forward(self,
                key: PRNGKeyArray,
                img: Array,
                mask: Array | None = None) -> Array:
        x = self.encoder(key, img, mask)
        return x 