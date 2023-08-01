import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Optional, Callable
from dataclasses import dataclass, field

@dataclass
class TransformerConfig:

    activation: Callable

    seed: int = field(default=2019612721831, metadata={"help": "seed for PRNG"})

    embed_dim: int = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
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
    attention_layers: list
    linear_layer1: eqx.nn.Linear
    linear_layer2: eqx.nn.Linear
    dropout_layer: eqx.nn.Dropout
    activation_dropout_layer: eqx.nn.Dropout

    # Feature TODO: add quantization
    # Feature TODO: Output full connected layer
    # Feature TODO: add full connected layer pruning

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        key = jax.random.PRNGKey(cfg.seed+1)
        self.activation = cfg.activation
        key, subkey = jax.random.split(key)
        self.attention_layers

        key,subkey
        self.linear_layer1 = eqx.nn.Linear(key=subkey, in_features=cfg.embed_dim, out_features=cfg.ffn_embed_dim)
        self.linear_layer2 = eqx.nn.Linear(key=subkey, in_features=cfg.ffn_embed_dim, out_features=cfg.embed_dim)
    
    def __call__(self,
                key: PRNGKeyArray,
                x: Array,
                encoder_padding_mask: Optional[Array],
                attention_mask: Optional[Array],):
        self.forward(key, x, encoder_padding_mask, attention_mask)
    
    def forward(self,
                key: PRNGKeyArray,
                x: Array,
                encoder_padding_mask: Optional[Array],
                attention_mask: Optional[Array],
                ):
        raise NotImplementedError


# @dataclass
# class TransformerEncoderConfig:


class TransformerEncoder(eqx.Module):

    # Feature TODO: add activation checkpointing
    # Feature TODO: add quantization
    # Feature TODO: add FSDP support

    token_embedding: eqx.nn.Embedding
    positional_embedding: eqx.nn.Embedding
    attention_blocks: list
    dropout_block: eqx.nn.Dropout
    feedforward_head: eqx.nn.Sequential

    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, tokens: Array) -> Array:
        return self.forward(tokens)
    
    def embed(self, tokens: Array) -> Array:
        raise NotImplementedError
    
    def forward(self, tokens: Array) -> Array:
        raise NotImplementedError
    
class TransformerDecoder(eqx.Module):

    def __init__(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
