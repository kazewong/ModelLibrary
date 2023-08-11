from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union

from kazeML.jax.common.Transformer import TransformerEncoder, TransformerConfig
from kazeML.jax.common.modules.EMA import EMAModule
from kazeML.jax.data2vec.feature_extractor import FeatureExtractor

@dataclass
class Data2VecConfig:

    transformer_encoder_config: TransformerConfig

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

class Data2Vec(eqx.Module):
    """
    Data2Vec model is a self-supervised learning model that learns
    representations from data.
    """

    feature_extractor: FeatureExtractor
    mask_embedding: Array
    encoder: TransformerEncoder
    ema: EMAModule
    mask_fraction: float = 0.6
    top_k_layer: int = 3

    def __init__(self,
                 key: PRNGKeyArray,
                 ):
        super().__init__()

        key, subkey = jax.random.split(key)

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def embed(self,
              x: Array,
              key: PRNGKeyArray) -> Array:
        feature = self.feature_extractor.extract_features(x)
        return self.encoder(feature, key, None)
    
    def forward_pair(self, 
                data: Array,
                key: PRNGKeyArray,
                ) -> tuple[Array, Array]:
        
        key, subkey = jax.random.split(key)
        x = self.feature_extractor.extract_features(data)
        mask = jnp.zeros(x.shape[0])
        mask_index = jax.random.choice(subkey, jnp.arange(x.shape[0]), shape=(int(x.shape[0]*self.mask_fraction),),replace=False)
        mask = mask.at[mask_index].set(1)[:,None]
        mask_x = x*(1-mask) + mask*self.mask_embedding

        pos_embedding = self.encoder.positional_embedding(x)
        x += pos_embedding
        mask_x += pos_embedding

        key, subkey = jax.random.split(key)
        x = self.encoder.dropout_block(x, key=subkey)

        key, subkey = jax.random.split(key)
        mask_x = self.encoder.dropout_block(mask_x, key=subkey)
        if self.encoder.embedding_layer_norm is not None:
            x = self.encoder.embedding_layer_norm(x)
            mask_x = self.encoder.embedding_layer_norm(mask_x)
        
        key, subkey = jax.random.split(key)
        y = self.ema.model.forward(x, subkey, None, layer_result=True)
        y = y[-self.top_k_layer:]
        y = jnp.mean(jnp.stack(y), axis=0)

        key, subkey = jax.random.split(key)
        x = self.encoder.forward(mask_x, subkey, None, layer_result=False)
        return x, y


    def d2v_loss(self,
                 data: Array,
                 key: PRNGKeyArray,
                 ) -> Array:
        student, teacher = self.forward_pair(data, key)
        return jnp.mean((student-teacher)**2)

    
    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str) -> None:
        return eqx.tree_deserialise_leaves(path+".eqx", self)