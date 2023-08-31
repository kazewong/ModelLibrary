from abc import ABC, abstractmethod
from typing import Callable
from jaxtyping import Array, PRNGKeyArray, Float, Int
import jax
import jax.numpy as jnp
import equinox as eqx


class FeatureExtractor(eqx.Module):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, data: Float[Array, "n_channel size"], key: PRNGKeyArray = None) -> Array:
        """

        The first dimension should be number of patches and the second dimension should be size of embedding.
        """
        pass


class ImageFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass


class SeriesFeatureExtractor(FeatureExtractor):
    layer: list[eqx.nn.Sequential]
    log_compression: bool
    skip_connections: bool
    residual_scale: float = 1.0

    def __init__(
        self,
        key: PRNGKeyArray,
        layer_spec: list[tuple[int, int, int]], # (in_channel, out_channel, kernel_size, stride)
        p_dropout: float = 0.0,
        affine_group_norm: bool = False,
        log_compression: bool = False,
        skip_connections: bool = False,
        residual_scale: float = 1.0,
        activation: Callable = jax.nn.gelu,
    ):
        super().__init__()

        self.layer = []
        for i in range(len(layer_spec) - 1):
            key, subkey = jax.random.split(key)
            self.layer.append(
                eqx.nn.Sequential(
                    [
                        eqx.nn.Conv1d(
                            layer_spec[i][0],
                            layer_spec[i + 1][0],
                            layer_spec[i][1],
                            layer_spec[i][2],
                            use_bias=False,
                            key=subkey,
                        ),
                        eqx.nn.Dropout(p_dropout),
                        eqx.nn.GroupNorm(
                            1,
                            layer_spec[i+1][0],
                            channelwise_affine=affine_group_norm,
                        ),
                        eqx.nn.Lambda(activation),
                    ]
                )
            )

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = residual_scale

    def extract_features(self, data: Float[Array, "n_channel size"], key: PRNGKeyArray = None) -> Array:
        residual = data
        subkey = key
        for i in range(len(self.layer)):
            if key != None: key, subkey = jax.random.split(key)
            data = self.layer[i](data, key = subkey)
            if self.skip_connections and data.shape[1] == residual.shape[1]:
                residual = residual[..., :: residual.shape[-1] // data.shape[-1]][
                    ..., : data.shape[-1]
                ]
                data = (data + residual) * self.residual_scale
                residual = data
        
        if self.log_compression:
            data = jnp.log(jnp.abs(data) + 1.)
        
        return data

class TextFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass
