from dataclasses import dataclass, fields
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, PRNGKeyArray, Float, PyTree

from kazeML.jax.common.Transformer import TransformerEncoder, TransformerConfig
from kazeML.jax.common.modules.EMA import EMAModule
from kazeML.jax.data2vec.feature_extractor import FeatureExtractor


@dataclass
class Data2VecConfig:
    transformer_encoder_config: TransformerConfig

    ema_decay: float = 0.99
    mask_fraction: float = 0.6
    top_k_layer = 3

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


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
        feature_extractor: FeatureExtractor,
        config: Data2VecConfig,
    ):
        super().__init__()

        key, subkey = jax.random.split(key)
        self.feature_extractor = feature_extractor
        self.encoder = TransformerEncoder(subkey, config.transformer_encoder_config)
        self.mask_embedding = jax.random.normal(subkey, (config.transformer_encoder_config.embed_dim,))
        self.ema = EMAModule(self.encoder, config.ema_decay)
        self.mask_fraction = config.mask_fraction
        self.top_k_layer = config.top_k_layer

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def embed(self, x: Array, key: PRNGKeyArray, deterministic: bool = True) -> Array:
        key, subkey = jax.random.split(key)
        if deterministic:
            extractor: FeatureExtractor = eqx.tree_inference(
                self.feature_extractor, value=True
            )
        else:
            extractor: FeatureExtractor = eqx.tree_inference(
                self.feature_extractor, value=True
            )
        feature: Array = extractor.extract_features(x, subkey)
        key, subkey = jax.random.split(key)
        return self.encoder(feature, subkey, None)

    def forward_pair(
        self,
        data: Float[Array, "n_channel n_size"],
        mask: Float[Array, "n_example n_token"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "n_example n_token n_embed"], Float[Array, "n_token n_embed"]]:
        key, subkey = jax.random.split(key)
        feature = self.feature_extractor.extract_features(data, subkey).T # to [n_token, n_embed]
        mask = mask[..., None]

        mask_feature = feature * (1 - mask) + mask * self.mask_embedding

        pos_embedding = self.encoder.positional_embedding(feature)
        feature += pos_embedding
        mask_feature = mask_feature + pos_embedding

        key, subkey = jax.random.split(key)
        feature = self.encoder.dropout_block(feature, key=subkey)
        key, *subkey = jax.random.split(key, len(mask) + 1)
        mask_feature = jax.vmap(lambda x, local_key: self.encoder.dropout_block(x, key=local_key), in_axes=(0,0))(mask_feature, jnp.array(subkey))

        if self.encoder.embedding_layer_norm is not None:
            feature = self.encoder.embedding_layer_norm(feature)
            mask_feature = jax.vmap(self.encoder.embedding_layer_norm)(mask_feature)  # type: ignore

        key, subkey = jax.random.split(key)
        target = self.ema.model.forward(feature, subkey, None, layer_result=True)
        target = target[-self.top_k_layer :]
        target = jax.lax.stop_gradient(jnp.mean(jnp.stack(target), axis=0))

        key, *subkey = jax.random.split(key, len(mask) + 1)
        prediction = jax.vmap(
            lambda x, local_key: self.encoder.forward(x, local_key, None, layer_result=False))(mask_feature,jnp.array(subkey))
        return jnp.array(prediction), target

    def d2v_loss(
        self,
        data: Float[Array, "n_channel n_size"],
        mask: Float[Array, "n_example n_channel n_size"],
        key: PRNGKeyArray,
    ) -> Array:
        student, teacher = self.forward_pair(data, mask, key)
        return jnp.mean((student - teacher) ** 2)

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> None:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)
