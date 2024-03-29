import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Float
from typing import Optional, Callable
from dataclasses import dataclass, field, fields

from kazeML.jax.common.modules.Embedding import EmbedBase, PositionalEmbedding


@dataclass
class TransformerConfig:
    activation: Callable

    max_length: int = field(
        default=512, metadata={"help": "max length of input sequence"}
    )

    embed_dim: int = field(default=512, metadata={"help": "embedding dimension"})
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )

    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm after embedding"}
    )

    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for feed-forward network"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    embedding_dropout: float = field(
        default=0.1, metadata={"help": "Embedding dropout probability"}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


class TransformerEncoderLayer(eqx.Module):

    """
    Transformer Encoder Layer
    Adopted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_layer.py
    """

    activation: Callable
    attention_layers: eqx.nn.MultiheadAttention
    attention_layernorm: eqx.nn.LayerNorm
    activation_dropout_layer: eqx.nn.Dropout
    dropout_layer: eqx.nn.Dropout
    final_layernorm: eqx.nn.LayerNorm
    linear_layer1: eqx.nn.Linear
    linear_layer2: eqx.nn.Linear

    normalize_before: bool = False

    # Feature TODO: add quantization
    # Feature TODO: Output full connected layer
    # Feature TODO: add full connected layer pruning
    # Feature TODO: Give support to attention masking

    def __init__(self, key: PRNGKeyArray, cfg: TransformerConfig):
        super().__init__()

        # Set activation
        self.activation = cfg.activation

        # Set attention layers
        key, subkey = jax.random.split(key)
        self.attention_layers = eqx.nn.MultiheadAttention(
            key=subkey,
            num_heads=cfg.attention_heads,
            query_size=cfg.embed_dim,
            dropout_p=cfg.attention_dropout,
        )

        self.dropout_layer = eqx.nn.Dropout(p=cfg.embedding_dropout)
        self.activation_dropout_layer = eqx.nn.Dropout(p=cfg.activation_dropout)

        # Set full connected layers
        key, subkey = jax.random.split(key)
        self.linear_layer1 = eqx.nn.Linear(
            key=subkey, in_features=cfg.embed_dim, out_features=cfg.ffn_embed_dim
        )

        key, subkey = jax.random.split(key)
        self.linear_layer2 = eqx.nn.Linear(
            key=subkey, in_features=cfg.ffn_embed_dim, out_features=cfg.embed_dim
        )

        # Set layernorm
        self.attention_layernorm = eqx.nn.LayerNorm(shape=cfg.embed_dim)
        self.final_layernorm = eqx.nn.LayerNorm(shape=cfg.embed_dim)

    def __call__(
        self,
        key: PRNGKeyArray,
        x: Array,
        encoder_padding_mask: Optional[Array],
        # attention_mask: Optional[Array],
    ):
        return self.forward(key, x, encoder_padding_mask)  # , attention_mask)

    def forward(
        self,
        key: PRNGKeyArray,
        x: Array,
        encoder_padding_mask: Optional[Array],
        # attention_mask: Optional[Array],
    ):
        residual = x
        if self.normalize_before:
            x = self.attention_layernorm(x)
        key, subkey = jax.random.split(key)
        x = self.attention_layers(x, x, x, encoder_padding_mask, key=subkey)
        key, subkey = jax.random.split(subkey)
        x = self.dropout_layer(x, key=subkey)
        x = residual + x
        if not self.normalize_before:
            x = self.attention_layernorm(x)

        x = self.activation(jax.vmap(self.linear_layer1)(x))
        key, subkey = jax.random.split(subkey)
        x = self.activation_dropout_layer(x, key=subkey)

        x = jax.vmap(self.linear_layer2)(x)
        key, subkey = jax.random.split(subkey)
        x = self.dropout_layer(x, key=subkey)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layernorm(x)
        return x


class TransformerEncoder(eqx.Module):
    # TODO: Make encoder operate on a sequences of embedding, i.e. take embedding out.
    # Feature TODO: add activation checkpointing
    # Feature TODO: add quantization
    # Feature TODO: add FSDP support

    positional_embedding: PositionalEmbedding
    embedding_layer_norm: eqx.nn.LayerNorm | None

    attention_blocks: list
    dropout_block: eqx.nn.Dropout
    layer_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        cfg: TransformerConfig,
    ):
        # Set positional embedding
        key, subkey = jax.random.split(key)
        self.positional_embedding = PositionalEmbedding(
            subkey, cfg.max_length, cfg.embed_dim
        )

        # Set embedding layer norm
        if cfg.layernorm_embedding:
            self.embedding_layer_norm = eqx.nn.LayerNorm(shape=cfg.embed_dim)
        else:
            self.embedding_layer_norm = None

        # Setup attention blocks
        self.attention_blocks = []
        for i in range(cfg.layers):
            key, subkey = jax.random.split(key)
            self.attention_blocks.append(TransformerEncoderLayer(subkey, cfg))

        # Setup dropout block
        self.dropout_block = eqx.nn.Dropout(p=cfg.embedding_dropout)

        # Setup layer norm
        self.layer_norm = eqx.nn.LayerNorm(shape=cfg.embed_dim)

    def __call__(
        self,
        embedding: Float[Array, "length embed_dim"],
        key: PRNGKeyArray,
        mask: Optional[Array] = None,
    ) -> Array | list:
        return self.forward(embedding, key, mask)

    def encode_position(self, embedding: Array, key: PRNGKeyArray) -> Array:
        embedding += self.positional_embedding(embedding)
        embedding = self.dropout_block(embedding, key=key)
        if self.embedding_layer_norm is not None:
            embedding = self.embedding_layer_norm(embedding)
        return embedding

    def forward(
        self,
        embedding: Float[Array, "length embed_dim"],
        key: PRNGKeyArray,
        mask: Optional[Array] = None,
        layer_result: bool = False,
    ) -> Array | list:
        """
        Forward pass of the transformer encoder.
        Input should already be embedded in an embedding space.
        """
        key, subkey = jax.random.split(key)
        x = embedding
        if layer_result:
            layer_results: list[Array] = []
            for block in self.attention_blocks:
                key, subkey = jax.random.split(key)
                x = block(subkey, x, mask)
                layer_results.append(x)
            return layer_results

        else:
            for block in self.attention_blocks:
                key, subkey = jax.random.split(key)
                x = block(subkey, x, mask)  # TODO: Need to figure how to better mask

            x = self.layer_norm(x)
            return x


class TransformerDecoder(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


def test_transformer_encoder_layer():
    # Define input and config
    input_shape = (4, 10, 32)
    cfg = TransformerConfig(
        activation=jax.nn.gelu,
        embed_dim=32,
    )
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, input_shape)

    # Initialize model
    key, subkey = jax.random.split(key)
    model = TransformerEncoder(subkey, cfg)

    # Test forward pass
    key, subkey = jax.random.split(key)
    output = model(x, subkey, None)
    if not isinstance(output, Array):
        output = output[0]
    assert output.shape == input_shape
