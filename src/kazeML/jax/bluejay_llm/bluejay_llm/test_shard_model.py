import math
from typing import Any
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.distributed import initialize
from jaxtyping import PRNGKeyArray
import equinox as eqx

from bluejay_llm.bluejay import GPT

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

def init_shard_linear(
        key: PRNGKeyArray, model: eqx.nn.Linear, mesh: jax.sharding.Mesh, sharding: jax.sharding.Sharding, n_devices: int = 1, shard_axis: int = 0
):
    lim = 1 / math.sqrt(model.in_features)

    key, subkey = jax.random.split(key)
    weight = init_shard_parameters(subkey, model.weight.shape, dtype, lim, mesh, sharding, n_devices, shard_axis)

    key, subkey = jax.random.split(key)
    bias = init_shard_parameters(subkey, model.bias.shape, dtype, lim, mesh, sharding, n_devices, shard_axis)

    model = eqx.tree_at(lambda m: m.weight, model, weight)
    model = eqx.tree_at(lambda m: m.bias, model, bias)

    return model

def init_shard_layer_norm(
        model: eqx.nn.LayerNorm, mesh: jax.sharding.Mesh, sharding: jax.sharding.Sharding, n_devices: int = 1
):

    sharded_length = model.shape[0]//n_devices
    sharded_shape = (sharded_length,)
    per_device_weight = jax.device_put(
        jnp.split(jnp.ones(sharded_shape), len(mesh.local_devices), axis=0),
        mesh.local_devices,
    )
    per_device_bias = jax.device_put(
        jnp.split(jnp.zeros(sharded_shape), len(mesh.local_devices), axis=0),
        mesh.local_devices,
    )
    weight = jax.make_array_from_single_device_arrays(
        model.shape, sharding, per_device_weight
    )
    bias = jax.make_array_from_single_device_arrays(
        model.shape, sharding, per_device_bias
    )

    model = eqx.tree_at(lambda m: m.weight, model, weight)
    model = eqx.tree_at(lambda m: m.bias, model, bias)

    return model

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
    model = eqx.filter_eval_shape(GPT, key=subkey)
    dtype = model.lm_head.weight.dtype

    # Initialize sharedblocks
    arrays, statics = eqx.partition(model.blocks, lambda x: isinstance(x, eqx.nn.Linear), is_leaf = lambda x: isinstance(x, eqx.nn.Linear))
    linears = jax.tree.leaves(arrays, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))
    new_layers = []
    for i in linears:
        key, subkey = jax.random.split(key)
        new_layers.append(init_shard_linear(subkey, i, mesh, sharding, n_processes))
    
    new_arrays = eqx.tree_at(lambda x: jax.tree.leaves(x, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)), arrays, new_layers)
    new_blocks = eqx.combine(new_arrays, statics, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))

    arrays, statics = eqx.partition(new_blocks, lambda x: isinstance(x, jax._src.api.ShapeDtypeStruct))
    masks = jax.tree.leaves(arrays)
    new_mask = []
    for i in masks:
        mask = -jnp.inf * jnp.invert(jnp.tril(jnp.ones((model.block_size, model.block_size), dtype=bool)))
        mask = jnp.nan_to_num(mask, posinf=jnp.inf, neginf=-jnp.inf)
        new_mask.append(mask)
    new_arrays = eqx.tree_at(lambda x: jax.tree.leaves(x), arrays, new_mask)
    new_blocks = eqx.combine(new_arrays, statics)
    model = eqx.tree_at(lambda x: x.blocks, model, new_blocks)

    # Initialize sharded layer norms
    arrays, statics = eqx.partition(model, lambda x: isinstance(x, eqx.nn.LayerNorm), is_leaf = lambda x: isinstance(x, eqx.nn.LayerNorm))
    layerNorms = jax.tree.leaves(arrays, is_leaf=lambda x: isinstance(x, eqx.nn.LayerNorm))
    new_layers = []
    for i in layerNorms:
        new_layers.append(init_shard_layer_norm(i, mesh, sharding, n_processes))
    new_arrays = eqx.tree_at(lambda x: jax.tree.leaves(x, is_leaf=lambda x: isinstance(x, eqx.nn.LayerNorm)), arrays, new_layers)
    model = eqx.combine(new_arrays, statics, is_leaf=lambda x: isinstance(x, eqx.nn.LayerNorm))

    # Initialize embedding
    key, subkey = jax.random.split(key)
    model = eqx.tree_at(lambda x: x.token_embedding, model, eqx.nn.Embedding(model.vocab_size, model.n_embed, key=subkey))
    model = eqx.tree_at(lambda x: x.position_embedding, model, eqx.nn.Embedding(model.block_size, model.n_embed, key=subkey))

    # Initialize lm_head
    key, subkey = jax.random.split(key)
    model = eqx.tree_at(lambda x: x.lm_head, model, eqx.nn.Linear(model.n_embed, model.vocab_size, key=subkey))

    data_local = jnp.ones(1024).astype(jnp.int32)

    f = eqx.filter_jit(model)
    result = f(data_local, key = jax.random.PRNGKey(0))

    print(result.devices())
    # value = process_allgather(result)
    # print(value)
