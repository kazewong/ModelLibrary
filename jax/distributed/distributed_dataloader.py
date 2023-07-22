import jax
import jax.numpy as jnp
import torchvision
import torch
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding, Mesh

import jax
import jax.numpy as jnp
from jax import sharding
from jax.sharding import 
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils

jax.distributed.initialize()
print(jax.process_count())
print(jax.devices())
print(jax.local_device_count())



normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)

