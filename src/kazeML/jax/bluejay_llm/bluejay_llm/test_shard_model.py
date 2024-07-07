import math
from typing import Any, Literal, Union
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.distributed import initialize
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._misc import default_floating_dtype

from jax.experimental.multihost_utils import process_allgather

def default_init(
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


class Linear_shard(eqx.nn.Linear):
    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        mesh: jax.sharding.Mesh,
        sharding: jax.sharding.Sharding,
        shard_axis: int = 0,
        n_devices: int = 1,
        use_bias: bool = True,
        dtype=None,        
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jax.random.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = default_init(wkey, wshape, dtype, lim, mesh, sharding, n_devices, shard_axis)
        bshape = (out_features_,)
        self.bias = default_init(bkey, bshape, dtype, lim, mesh, sharding, n_devices, shard_axis) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


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

    model = Linear_shard(24, 24*4, mesh=mesh, sharding=sharding, n_devices=n_processes, key = jax.random.PRNGKey(0))
    data_local = jnp.ones(24)

    f = eqx.filter_jit(model)
    result = f(data_local)

    print(result.devices())
    value = process_allgather(result)
    print(value)
