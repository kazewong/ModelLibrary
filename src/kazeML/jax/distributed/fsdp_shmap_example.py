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

model = eqx.nn.Linear(784, 16, key=jax.random.PRNGKey(0))

devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, ('batch',))

data = jnp.ones((8, 784))
data_sharded = jax.device_put(data, NamedSharding(mesh, P('batch', )))
model_sharded = jax.device_put(model, NamedSharding(mesh, P('batch', )))