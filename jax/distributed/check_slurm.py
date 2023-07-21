# Do salloc -p gpu --nodes N --ntasks-per-node G --gpus-per-node=G --cpus-per-task C -C a100
# Note that ntasks-per-node and gpus-per-node must be the same. Otherwise jax won't recognize the number of devices.
# Then srun bash "submit_script", which load the environment correctly.
# BURN BABY BURN!
import os
import time

os.system("echo SLURM_ID: $SLURM_JOB_ID")
os.system("echo SLURM_NTASKS: $SLURM_NTASKS")
os.system("echo SLURM_NODELIST: $SLURM_NODELIST")
os.system("echo SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST")
os.system("echo SLURM_STEP_GPUS: $SLURM_STEP_GPUS")
os.system("echo SLURM_GPUS: $SLURM_GPUS")

import jax
import jax.numpy as jnp
from jax import sharding
from jax.sharding import Mesh
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils

jax.distributed.initialize()
print(jax.process_count())
print(jax.devices())
print(jax.local_device_count())

sharding = sharding.PositionalSharding(mesh_utils.create_device_mesh((4,2)))

# devices = np.array(jax.devices()).reshape(4,2)
# global_mesh = Mesh(devices, ('x','y'))
# shard = sharding.NamedSharding(global_mesh, P("x","y"))

xs = jax.numpy.ones(jax.local_device_count())+ jax.process_index()
# print(jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs))

# x_size = 2**14
# y_size = 2**14

# x = jax.random.normal(jax.random.PRNGKey(0), (x_size, y_size))

# y = jax.device_put(x)

# print(x.devices(), y.devices())
# run_time = time.time()
# z = y@y
# print(time.time()-run_time)

# with global_mesh:
#     out = pjit(lambda x, y : x@y)(inp, inp)
#     print(out)
# jax.debug.visualize_array_sharding(x)


