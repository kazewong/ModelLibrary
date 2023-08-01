import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Optional, Callable
from dataclasses import dataclass, field

from kazeML.jax.common.modules.Embedding import EmbedBase, PositionalEmbedding

@dataclass
class TransformerConfig:

    activation: Callable

    seed: int = field(default=2019612721831, metadata={"help": "seed for PRNG"})

    max_length: int = field(default=512, metadata={"help": "max length of input sequence"})

    embed_dim: int = field(
        default=512, metadata={"help": "embedding dimension"}
    )
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
    dropout: float = field(default=0.1, metadata={"help": "Embedding dropout probability"})
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

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        # Set activation
        key = jax.random.PRNGKey(cfg.seed+1201591029)
        self.activation = cfg.activation
        
        # Set attention layers
        key, subkey = jax.random.split(key)
        self.attention_layers = eqx.nn.MultiheadAttention(key=subkey,
                                                        num_heads=cfg.attention_heads,    
                                                        query_size=cfg.embed_dim,
                                                        dropout_p=cfg.attention_dropout)

        self.dropout_layer = eqx.nn.Dropout(p=cfg.dropout)
        self.activation_dropout_layer = eqx.nn.Dropout(p=cfg.activation_dropout)

        # Set full connected layers
        key, subkey = jax.random.split(key)
        self.linear_layer1 = eqx.nn.Linear(key=subkey, in_features=cfg.embed_dim, out_features=cfg.ffn_embed_dim)

        key, subkey = jax.random.split(key)
        self.linear_layer2 = eqx.nn.Linear(key=subkey, in_features=cfg.ffn_embed_dim, out_features=cfg.embed_dim)

        # Set layernorm
        self.attention_layernorm = eqx.nn.LayerNorm(shape=cfg.embed_dim)
        self.final_layernorm = eqx.nn.LayerNorm(shape=cfg.embed_dim)

    def __call__(self,
                key: PRNGKeyArray,
                x: Array,
                encoder_padding_mask: Optional[Array],
                # attention_mask: Optional[Array],
                ):
        self.forward(key, x, encoder_padding_mask)#, attention_mask)
    
    def forward(self,
                key: PRNGKeyArray,
                x: Array,
                encoder_padding_mask: Optional[Array],
                # attention_mask: Optional[Array],
                ):

        residual = x
        if self.normalize_before: x = self.attention_layernorm(x)
        x = self.attention_layers(x, x, x, encoder_padding_mask, key=key)
        x = self.dropout_layer(x)
        x = residual + x
        if not self.normalize_before: x = self.attention_layernorm(x)

        x = self.activation(self.linear_layer1(x))
        x = self.activation_dropout_layer(x)

        x = self.linear_layer2(x)
        x = self.dropout_layer(x)
        x = residual + x
        if not self.normalize_before: x = self.final_layernorm(x)
        return x

class TransformerEncoder(eqx.Module):

    # TODO: Make encoder operate on a sequences of embedding, i.e. take embedding out.
    # Feature TODO: add activation checkpointing
    # Feature TODO: add quantization
    # Feature TODO: add FSDP support

    token_embedding: EmbedBase
    positional_embedding: PositionalEmbedding
    embedding_layer_norm: eqx.nn.LayerNorm | None

    attention_blocks: list
    dropout_block: eqx.nn.Dropout
    layer_norm: eqx.nn.LayerNorm

    def __init__(self,
                cfg: TransformerConfig,
                embed_tokens: EmbedBase,):
        key = jax.random.PRNGKey(cfg.seed+1029571204)

        # Set token embedding
        self.token_embedding = embed_tokens

        # Set positional embedding
        key, subkey = jax.random.split(key)
        self.positional_embedding = PositionalEmbedding(subkey, cfg.max_length, cfg.embed_dim)

        # Set embedding layer norm
        if cfg.layernorm_embedding: self.embedding_layer_norm = eqx.nn.LayerNorm(shape=cfg.embed_dim)

        # Setup attention blocks
        self.attention_blocks = []
        for i in range(cfg.layers):
            key, subkey = jax.random.split(key)
            self.attention_blocks.append(TransformerEncoderLayer(cfg))

        # Setup dropout block
        self.dropout_block = eqx.nn.Dropout(p=cfg.dropout)

        # Setup layer norm
        self.layer_norm = eqx.nn.LayerNorm(shape=cfg.embed_dim)
    
    def __call__(self,
                key: PRNGKeyArray,
                tokens: Array,
                mask: Optional[Array] = None,) -> Array:
        return self.forward(key, tokens, mask)
    
    def embed(self, tokens: Array) -> Array:
        embedding = self.token_embedding(tokens) + self.positional_embedding(tokens)
        embedding = self.dropout_block(embedding)
        if self.embedding_layer_norm is not None: embedding = self.embedding_layer_norm(embedding)
        return embedding
    
    def forward(self,
                key: PRNGKeyArray,
                tokens: Array,
                mask: Optional[Array] = None,
                ) -> Array:
        x = self.embed(tokens)
        for block in self.attention_blocks:
            key, subkey = jax.random.split(key)
            x = block(subkey, x, mask) # TODO: Need to figure how to better mask

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
    cfg = TransformerConfig(activation=jax.nn.gelu, embed_dim = 32, )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, input_shape)

    # Initialize model
    model = TransformerEncoder(cfg)

    # Test forward pass
    output = model(key, x, None)
    assert output.shape == input_shape