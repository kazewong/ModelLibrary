import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16' # Use 8 CPU devices

from functools import partial

import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import equinox as eqx

model = eqx.nn.Linear(784, 16, key=jax.random.PRNGKey(0))

devices = mesh_utils.create_device_mesh((16,))
mesh = Mesh(devices, ('batch',))

data = jnp.ones((16*1000, 784))
data_sharded = jax.device_put(data, NamedSharding(mesh, P('batch', )))
model_sharded = jax.device_put(model, NamedSharding(mesh, P('batch', )))

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(P('batch'), P('batch')), out_specs=P('batch'))
def linear_fsdp(model, x):
    # return jax.vmap(model)(x)
    return jax.lax.all_gather(jax.vmap(model)(x), 'batch', tiled=True, axis=1)

c = linear_fsdp(model_sharded, data_sharded)
d = jax.vmap(model)(data)

print(jnp.allclose(c, d))