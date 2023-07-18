from sde_score import ScordBasedSDE
from common.Unet import Unet
import jax
import jax.numpy as jnp

seed = 0

unet = Unet(2, [1,4,8,16], jax.random.PRNGKey(seed))
sde = ScordBasedSDE(unet, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0)

test_data = jnp.zeros((1, 32, 32))