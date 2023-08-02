# Vision Model of Data2Vec
# Inspired by Fairseq's Data2Vec vision Model

import equinox as eqx
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from kazeML.jax.common.Transformer import TransformerConfig, TransformerEncoder
from kazeML.jax.common.modules.EMA import EMAModule

from kazeML.jax.common.modules.Embedding import PatchEmbedding


@dataclass
class Data2VecVisionConfig:

    transformer_decoder_config: TransformerConfig

    layer_scale_init_value: float = 1e-4
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    shared_rel_pos_bias: bool = True

    drop_path: float = 0.1
    attention_dropout: float = 0.0
    embed_dim: int = 768

    ema_decay: float = 0.99
    mask_fraction: float = 0.6
    top_k_layer = 3


class Data2VecVision(eqx.Module):

    class_embedding: Array
    mask_embedding: Array
    encoder: TransformerEncoder
    final_projection: eqx.nn.Linear
    ema: EMAModule
    mask_fraction: float = 0.6
    top_k_layer: int = 3

    def __init__(self, key: PRNGKeyArray,
                cfg: Data2VecVisionConfig):
        super().__init__()

        key, subkey = jax.random.split(key)
        patch_embed = PatchEmbedding(key=subkey,
                                      patch_size=cfg.patch_size,
                                      img_size=cfg.image_size,
                                      in_channels=cfg.in_channels,
                                      embed_dim=cfg.embed_dim)
        
        key, subkey = jax.random.split(key)
        self.class_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        key, subkey = jax.random.split(key)
        self.mask_embedding = jax.random.truncated_normal(key=subkey, lower=-2.0, upper=2.0, shape=(cfg.embed_dim,)) + 0.02

        key, subkey = jax.random.split(key)
        self.encoder = TransformerEncoder(subkey, cfg.transformer_decoder_config, patch_embed)
        self.ema = EMAModule(self.encoder, cfg.ema_decay)

        key, subkey = jax.random.split(key)
        self.final_projection = eqx.nn.Linear(key=subkey, in_features=cfg.embed_dim, out_features=cfg.embed_dim)

        self.mask_fraction = cfg.mask_fraction
        self.top_k_layer = cfg.top_k_layer

    def loss(self,
            img: Array,
            key: PRNGKeyArray,) -> Array:

        # Create embedding and masks
        key, subkey = jax.random.split(key)
        x = self.encoder.token_embedding(img)
        mask = jnp.zeros(x.shape[0])
        mask_index = jax.random.choice(subkey, jnp.arange(x.shape[0]), shape=(int(x.shape[0]*self.mask_fraction),),replace=False)
        mask = mask.at[mask_index].set(1)[:,None]
        mask_x = x*(1-mask) + mask*self.class_embedding

        x += self.encoder.positional_embedding(x)
        mask_x += self.encoder.positional_embedding(mask_x)

        key, subkey = jax.random.split(key)
        x = self.encoder.dropout_block(x, key=subkey)

        key, subkey = jax.random.split(key)
        mask_x = self.encoder.dropout_block(mask_x, key=subkey)
        if self.encoder.embedding_layer_norm is not None:
            x = self.encoder.embedding_layer_norm(x)
            mask_x = self.encoder.embedding_layer_norm(mask_x)
        
        key, subkey = jax.random.split(key)
        y = self.ema.model.forward(subkey, x, None, layer_result=True)
        y = y[-self.top_k_layer:]
        y = jnp.mean(jnp.stack(y), axis=0)

        key, subkey = jax.random.split(key)
        x = self.encoder.forward(subkey, mask_x, None, layer_result=False)
        return jnp.mean((x-y)**2)

    def encode(self,
               img: Array,
               key: PRNGKeyArray,) -> Array:
        x = self.encoder(key, img, None)
        x = self.final_projection(x)
        return x
