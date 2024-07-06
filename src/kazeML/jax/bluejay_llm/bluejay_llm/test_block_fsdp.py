import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

from functools import partial

import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')


from bluejay_llm.bluejay import Block

import equinox as eqx

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import process_allgather


model = Block(key=jax.random.PRNGKey(0))

devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, ('batch',))

data = jnp.ones((8*1000, 768))
data_sharded = jax.device_put(data, NamedSharding(mesh, P('batch', )))
array, statics = eqx.partition(model, eqx.is_array)
array = jax.device_put(array, NamedSharding(mesh, P('batch', )))
model_sharded = eqx.combine(array, statics)

# @jax.jit
# @partial(shard_map, mesh=mesh, in_specs=(P('batch'), P('batch')), out_specs=P('batch'))
# def linear_fsdp(model, x):
