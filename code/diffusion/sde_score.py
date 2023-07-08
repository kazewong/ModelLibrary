import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

class ScordBasedSDE(eqx.Module):

    blocks: list

    def __init__(self,
                 blocks: list,
                 key: PRNGKeyArray,
                 ):
        self.blocks = blocks

    def __call__(self):
        pass

    def score(self):
        pass

    def sample(self):
        pass