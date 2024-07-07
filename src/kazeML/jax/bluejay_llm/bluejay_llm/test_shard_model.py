import math
from typing import Any
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.distributed import initialize
from jaxtyping import PRNGKeyArray
import equinox as eqx


def init_shard_parameters(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float, mesh: jax.sharding.Mesh, sharding: jax.sharding.Sharding, n_devices: int = 1, shard_axis: int = 0
) -> jax.Array:
    sharded_length = shape[shard_axis]//n_devices
    sharded_shape = shape[:shard_axis] + (sharded_length,) + shape[shard_axis+1:]
    per_device_array = jax.device_put(
        jnp.split(
            jax.random.uniform(key, sharded_shape, dtype, minval=-lim, maxval=lim)
            , len(mesh.local_devices), axis=shard_axis
        ),
        mesh.local_devices,
    )
    return jax.make_array_from_single_device_arrays(
        shape, sharding, per_device_array
    )



if __name__ == "__main__":
    initialize()
    if jax.process_index() == 0:
        print("Total number of process: " + str(jax.process_count()))

    n_processes = jax.process_count()
    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, ("batch"))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(
            ("batch"),
        )
    )

    key = jax.random.PRNGKey(10293801)
    key, subkey = jax.random.split(key)

    # This shards the model across N process, so 5 40GBs GPUs should be able to host it
    model = eqx.filter_eval_shape(eqx.nn.Linear, in_features=1_000, out_features=1_000_000*24, key=subkey)
    dtype = model.weight.dtype
    lim = 1 / math.sqrt(model.in_features)

    key, subkey = jax.random.split(key)
    weight = init_shard_parameters(key, model.weight.shape, dtype, lim, mesh, sharding, n_processes, 0)

    key, subkey = jax.random.split(key)
    bias = init_shard_parameters(key, model.bias.shape, dtype, lim, mesh, sharding, n_processes, 0)

    model = eqx.tree_at(lambda m: m.weight, model, weight)
    model = eqx.tree_at(lambda m: m.bias, model, bias)

    # This requests 192GB of RAMs, and it should fail on a single process
    # model = eqx.nn.Linear(1000, 1_000_000*24, key = jax.random.PRNGKey(0))

    data_local = jnp.ones(1000)

    f = eqx.filter_jit(model)
    result = f(data_local)

    print(result.devices())
    # value = process_allgather(result)
    # print(value)
