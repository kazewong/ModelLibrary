import jax
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
import numpy as np

jax.distributed.initialize()
print(jax.process_count())
print(jax.devices())
print(jax.local_device_count())

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 200
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 4
TIME_FEATURE = 128
AUTOENCODER_EMBED_DIM = 256

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "/mnt/home/wwong/MLProject/ModelLibrary/data/MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)

print("Creating sampler")

sampler = DistributedSampler(train_dataset,
                            num_replicas=jax.process_count(),
                            rank=jax.process_index(),
                            shuffle=True,
                            seed=SEED)

print("Creating dataloader")

trainloader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        sampler=sampler,
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True)

global_mesh = jax.sharding.Mesh(np.array(jax.devices()), ('b'))

with global_mesh:
    for i, (x, y) in enumerate(trainloader):
        y = jnp.array(y)
        y = jax.device_put(y)[None,:]*jax.process_index()
        if i==0:
            print(y.shape, y.devices())
            z = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(y)
            print(z)