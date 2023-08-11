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
        self.feature_extractor.extract_features(x)
        x = self.encoder
    
    def forward(self, 
                source: Array,
                target: Array,
                padding_mask: Array,
                mask: bool = True,
                ) -> Array:
        
        # Feature extraction

        

        # 

        # d2v loss



        raise NotImplementedError

    def d2v_loss(self,
                 data: Array,
                 key: PRNGKeyArray,
                 ) -> Array:
        raise NotImplementedError
    
    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str) -> None:
        return eqx.tree_deserialise_leaves(path+".eqx", self)