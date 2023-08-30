from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, PRNGKeyArray, Float, PyTree
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
    ema: EMAModule[TransformerEncoder]
    mask_fraction: float = 0.6
    top_k_layer: int = 3

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        super().__init__()

        key, subkey = jax.random.split(key)

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def embed(self, x: Array, key: PRNGKeyArray, deterministic: bool = True) -> Array:
        key, subkey = jax.random.split(key)
        if deterministic:
            extractor: FeatureExtractor = eqx.tree_inference(self.feature_extractor, value=True)
        else:
            extractor: FeatureExtractor = eqx.tree_inference(self.feature_extractor, value=True)
        feature: Array = extractor.extract_features(x, subkey)
        key, subkey = jax.random.split(key)
        return self.encoder(feature, subkey, None)

    def forward_pair(
        self,
        data: Float[Array, "n_channel n_size"],
        mask: list[Float[Array, "n_example n_channel n_size"]],
        key: PRNGKeyArray,
    ) -> tuple[PyTree, Array]:
        key, subkey = jax.random.split(key)
        feature = self.feature_extractor.extract_features(data, subkey)

        mask_feature = tree_map(lambda x: feature * (1 - x) + x * self.mask_embedding, mask)

        pos_embedding = self.encoder.positional_embedding(feature)
        feature += pos_embedding
        mask_feature = tree_map(lambda x: x + pos_embedding, mask_feature)

        key, subkey = jax.random.split(key)
        feature = self.encoder.dropout_block(feature, key=subkey)
        key, *subkey = jax.random.split(key,len(mask)+1)
        mask_feature = tree_map(lambda x, local_key: self.encoder.dropout_block(x, key=local_key), mask_feature, subkey)

        if self.encoder.embedding_layer_norm is not None:
            feature = self.encoder.embedding_layer_norm(feature)
            mask_feature = tree_map(lambda x: self.encoder.embedding_layer_norm(x), mask_feature) # type: ignore

        key, subkey = jax.random.split(key)
        target = self.ema.model.forward(feature, subkey, None, layer_result=True)
        target = target[-self.top_k_layer :]
        target = jnp.mean(jnp.stack(target), axis=0)

        key, *subkey = jax.random.split(key, len(mask) + 1)
        prediction = tree_map(
            lambda x, local_key: self.ema.model.forward(x, local_key, None, layer_result=False),
            mask_feature,
            subkey,
        )
        return prediction, target

    def d2v_loss(
        self,
        data: Float[Array, "n_channel n_size"],
        mask: list[Float[Array, "n_example n_channel n_size"]],
        key: PRNGKeyArray,
    ) -> Array:
        student, teacher = self.forward_pair(data, mask, key)
        return jnp.mean((student - teacher) ** 2)

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> None:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)
