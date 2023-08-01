import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class TransformerConfig:

    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for feed-forward network"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    layers_to_keep: Optional[list[int]] = field(
        default=None, metadata={"help": "which layers to *keep* when pruning"}
    )

    xformers_att_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "config for xFormers attention, defined in xformers.components.attention.AttentionConfig"
        },
    )

    
class TransformerEncoderLayer(eqx.Module):

    # Feature TODO: add quantization

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
    
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
