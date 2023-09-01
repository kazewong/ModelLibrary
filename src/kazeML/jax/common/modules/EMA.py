import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable, Union, Generic, TypeVar

T = TypeVar("T")

def replace(model, update):
    if update is None:
        return model
    else:
        return update

class EMAModule(eqx.Module, Generic[T]):
    """
    Exponential Moving Average tracking of a model's parameters.
    """

    model: T
    decay: float = 0.99
    log_norms: bool = False

    def __init__(self,
                model: T,
                decay: float = 0.99,
                log_norms: bool = False):
        self.model = model 
        self.decay = decay
        self.log_norms = log_norms

    def __call__(self, new_model: eqx.Module) -> T:
        return self.step(new_model)

    def set_decay(self, decay: float) -> eqx.Module:
        return eqx.tree_at(lambda x: x.decay, self, decay)

    def set_model(self, pyTree: PyTree) -> T:
        new_tree = jax.tree_util.tree_map(replace, jax.tree_util.tree_leaves(self.model), pyTree)
        return eqx.tree_at(lambda x: jax.tree_util.tree_leaves(x), self.model, new_tree)

    def step(self, new_model: eqx.Module) -> T:
        decay = self.decay
        lr = 1 - decay
        ema_params = eqx.filter(jax.tree_util.tree_leaves(self.model), eqx.is_array)
        new_params = eqx.filter(jax.tree_util.tree_leaves(new_model), eqx.is_array)
        new_ema_tree = jax.tree_util.tree_map(lambda x,y: x*decay+y*lr, ema_params, new_params)
        return self.set_model(new_ema_tree)