import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from typing import Callable
from kazeML.jax.common.modules.Embedding import PositionalEmbedding

class AttentionBlock(eqx.Module):

    attention: eqx.nn.MultiheadAttention
    linear_layer: list
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self,
                key: PRNGKeyArray,
                embed_dim: int,
                hidden_dim: int,
                num_heads: int,
                dropout_prob: float = 0.0,
                ):
        self.attention = eqx.nn.MultiheadAttention(num_heads, embed_dim, key=key, dropout_p=dropout_prob)
        self.linear_layer = [eqx.nn.Linear(embed_dim, hidden_dim, key=key),
             eqx.nn.Lambda(jax.nn.gelu),
             eqx.nn.Dropout(dropout_prob),
             eqx.nn.Linear(hidden_dim, embed_dim, key=key)]

        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.dropout = eqx.nn.Dropout(dropout_prob)


    def __call__(self, x: Array, key: PRNGKeyArray, inference: bool = False) -> Array:
        norm_x = self.layer_norm1(x)
        key, subkey = jax.random.split(key)
        attention_out = self.attention(norm_x, norm_x, norm_x, key=subkey, inference=inference)
        
        key, subkey = jax.random.split(key)
        x = x + self.dropout(attention_out, key=key, inference=inference)
        linear_out = self.layer_norm2(x)
        linear_out = jax.vmap(self.linear_layer[0])(linear_out)
        linear_out = self.linear_layer[1](linear_out)
        key, subkey = jax.random.split(key)
        linear_out = self.linear_layer[2](linear_out, key=subkey, inference=inference)
        linear_out = jax.vmap(self.linear_layer[3])(linear_out)
        key, subkey = jax.random.split(key)
        output = x + self.dropout(linear_out, key=subkey, inference=inference)
        return output

class VIT(eqx.Module):

    linear_projector: eqx.nn.Linear
    attention_blocks: list
    dropout_block: eqx.nn.Dropout
    feedforward_head: eqx.nn.Sequential
    patch_size: int
    position_embedding: PositionalEmbedding
    class_token: Array

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
        self.patch_size = patch_size

        key, subkey = jax.random.split(key)
        self.linear_projector = eqx.nn.Linear(patch_size*patch_size*num_channels, embed_dim, key=subkey)

        self.attention_blocks = []
        for i in range(num_layers):
            key, subkey = jax.random.split(key)
            self.attention_blocks.append(AttentionBlock(subkey, embed_dim, hidden_dim, num_heads, dropout_prob) for _ in range(num_layers))

        self.dropout_block = eqx.nn.Dropout(dropout_prob)

        key, subkey = jax.random.split(key)
        self.feedforward_head = eqx.nn.Sequential(
            [eqx.nn.LayerNorm(hidden_dim),
             eqx.nn.Linear(hidden_dim, num_classes, key=subkey)
            ]
        )

        key, subkey = jax.random.split(key)
        self.class_token = jax.random.normal(key, (1, embed_dim))

        key, subkey = jax.random.split(key)
        self.position_embedding = PositionalEmbedding(max_len=num_patches, embed_dim=embed_dim, key=subkey)

    def __call__(self, x: Array, key: PRNGKeyArray, inference: bool = False) -> Array:
        """
        Args:
            x (Array): Input image of shape (Channels, Height, Width)
            key (PRNGKeyArray): Random key for dropout
            train (bool, optional): Whether to apply dropout or not. Defaults to True.

        """
        x = self.image_to_patch(x)
        N, _ = x.shape
        x = jax.vmap(self.linear_projector)(x)

        x = jnp.concatenate([self.class_token, x], axis=0)
        x = x + self.position_embedding(x)

        key, subkey = jax.random.split(key)
        x = self.dropout_block(x, key=subkey, inference=inference)
        for attention_block in self.attention_blocks:
            key, subkey = jax.random.split(key)
            x = attention_block(x, key=subkey, inference=inference)

        class_token = x[:, 0]
        out = jax.vmap(self.feedforward_head)(class_token)
        return out

    def image_to_patch(self, x: Array, flatten_channels: bool = True) -> Array:
        """
        Args:
            x (Array): Input image of shape (Channels, Height, Width)

        Returns:
            Array: Image patches of shape (Height*Width/(Patch_Size*Patch_Size), Patch_Size*Patch_Size*Num_Channels)
        """
        C, H, W = x.shape
        x = x.reshape(C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.transpose(1, 3, 0, 2, 4)
        x = x.reshape(-1, C, self.patch_size, self.patch_size)
        if flatten_channels:
            x = x.reshape(x.shape[0], self.patch_size*self.patch_size*C)
        return x

def test_vit():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (5, 3, 32, 32))
    model = VIT(key, 128, 512, 4, 3, 6, 10, 4, 64, 0.1)
    key, subkey = jax.random.split(key)
    model(x[0], subkey)
    