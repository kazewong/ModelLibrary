import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union

class TransformerEncoder(eqx.Module):

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