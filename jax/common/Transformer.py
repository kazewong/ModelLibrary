import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union

class TransformerEncoder(eqx.Module):

    def __init__(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
class TransformerDecoder(eqx.Module):

    def __init__(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError