import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from typing import Callable

class VIT(eqx.Module):

    linear_projector: eqx.Module
    attention_blocks: list[eqx.nn.MultiheadAttention]
    dropout_block: eqx.nn.Dropout
    feedforward_head: eqx.Module




    def __init__(self,
                key: PRNGKeyArray,
                embed_dim : int,     # Dimensionality of input and attention feature vectors
                hidden_dim : int,    # Dimensionality of hidden layer in feed-forward network
                num_heads : int,     # Number of heads to use in the Multi-Head Attention block
                num_channels : int,  # Number of channels of the input (3 for RGB)
                num_layers : int,    # Number of layers to use in the Transformer
                num_classes : int,   # Number of classes to predict
                patch_size : int,    # Number of pixels that the patches have per dimension
                num_patches : int,   # Maximum number of patches an image can have
                dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network
                ):
        self.dropout_block = eqx.nn.Dropout(dropout_prob)
        key, subkey = jax.random.split(key)
        self.feedforward_head = eqx.nn.Sequential(
            [eqx.nn.LayerNorm(hidden_dim),
             eqx.nn.Linear(hidden_dim, num_classes, key=subkey)
            ]
        )

    def __call__(self, x: Array, key: PRNGKeyArray, train: bool = True) -> Array:
        raise NotImplementedError