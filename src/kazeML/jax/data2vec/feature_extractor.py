from abc import ABC, abstractmethod
from typing import Callable
from jaxtyping import Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx


class FeatureExtractor(eqx.Module):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, data: Array) -> Array:
        """

        The first dimension should be number of patches and the second dimension should be size of embedding.
        """
        pass

    @abstractmethod
    def embed(
        self,
        data: Array,
    ) -> Array:
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
    residual_connections: float = 1.0

    def __init__(
        self,
        key: PRNGKeyArray,
        num_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        p_dropout: float = 0.0,
        affine_group_norm: bool = False,
        log_compression: bool = False,
        skip_connections: bool = False,
        residual_connections: float = 1.0,
        activation: Callable = jax.nn.gelu,
    ):
        super().__init__()

        self.layer = []
        for i in range(num_layers):
            key, subkey = jax.random.split(key)
            self.layer.append(
                eqx.nn.Sequential(
                    [
                        eqx.nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size,
                            stride,
                            use_bias=False,
                            key=subkey
                        ),
                        eqx.nn.Dropout(p_dropout),
                        eqx.nn.GroupNorm(
                            1,
                            num_channels,
                            affine=affine_group_norm,
                        ),
                        eqx.nn.Lambda(activation)
                    ]
                )
            )

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_connections = residual_connections


    def extract_features(self, data):
        pass


class TextFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass
