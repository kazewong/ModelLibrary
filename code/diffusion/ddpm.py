import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

class DDPM(eqx.Module):
    