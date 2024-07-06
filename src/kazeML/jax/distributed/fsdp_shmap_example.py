import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

from functools import partial

import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import equinox as eqx

model = eqx.nn.Linear(784, 8, key=jax.random.PRNGKey(0))

devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, ('batch',))

data = jnp.ones((784))
data_sharded = jax.device_put(data, NamedSharding(mesh, P('batch', )))
arrays, statics = eqx.partition(model, lambda x: eqx.is_array(x) and x.ndim == 2)
arrays = jax.device_put(arrays, NamedSharding(mesh, P(None, 'batch')))
model_sharded = eqx.combine(arrays, statics)
arrays, statics = eqx.partition(model_sharded, lambda x: eqx.is_array(x) and x.ndim == 1)
arrays = jax.device_put(arrays, NamedSharding(mesh, P('batch')))
model_sharded = eqx.combine(arrays, statics)

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(P(None), P('batch')), out_specs=P('batch'))
def linear_fsdp(model, x):
    return model(x)
    # return jax.lax.all_gather(model(x), 'batch', tiled=True, axis=0)

c = linear_fsdp(model_sharded, data_sharded)
d = jax.vmap(model)(data)

print(jnp.allclose(c, d))