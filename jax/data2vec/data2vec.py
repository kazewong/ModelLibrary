import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Union

class Data2Vec(eqx.Module):
    """
    Data2Vec model is a self-supervised learning model that learns
    representations from data.
    """

    def __init__(self):
        pass

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def extract_feature(self, x: Array) -> Array:
        raise NotImplementedError
    
    def forward(self, 
                source: Array,
                target: Array,
                padding_mask: Array,
                mask: bool = True,
                ) -> Array:
        
        # Feature extraction

        # 

        # d2v loss



        raise NotImplementedError

    def loss(self, source: Array, target: Array) -> Array:
        """
        Compute the loss function. 
        The goal of the data2vec loss function is to make the student model
        mimic the teacher model.

        Args:
            source (Array): teacher representation
            target (Array): student representation
        """
        raise NotImplementedError