
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, Array, Float
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import optax
import equinox as eqx
import jax
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
import numpy as np

from kazeML.jax.common.Transformer import TransformerConfig
from kazeML.jax.data2vec.data2vec_vision import Data2VecVision, Data2VecVisionConfig
from kazeML.jax.common.modules.EMA import EMAModule

transformer_config = TransformerConfig(eqx.nn.Lambda(jax.nn.gelu), embed_dim=64)
D2V_cfg = Data2VecVisionConfig(transformer_config, image_size=28, patch_size=4, in_channels=1,
                               embed_dim=64)


BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 200
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 12
TIME_FEATURE = 128
AUTOENCODER_EMBED_DIM = 256


initialize()

if jax.process_index() == 0:
    print(jax.process_count())
    print(jax.devices())
    print(jax.local_device_count())

model = Data2VecVision(jax.random.PRNGKey(0),D2V_cfg)

optimizer = optax.adam(LEARNING_RATE)

normalize_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "./data/MNIST",
    train=True,
    download=True,
    transform=normalize_data,
)
test_dataset = torchvision.datasets.MNIST(
    "./data/MNIST",
    train=False,
    download=True,
    transform=normalize_data,
)

train_sampler = DistributedSampler(train_dataset,
                            num_replicas=jax.process_count(),
                            rank=jax.process_index(),
                            shuffle=True,
                            seed=SEED)

test_sampler = DistributedSampler(test_dataset,
                            num_replicas=jax.process_count(),
                            rank=jax.process_index(),
                            shuffle=True,
                            seed=SEED)


trainloader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        sampler=train_sampler,
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True)

testloader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        sampler=test_sampler,
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True)


trainloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
testloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


def train(
    model: Data2VecVision,
    trainloader: DataLoader,
    testloader: DataLoader,
    key: PRNGKeyArray,
    steps: int = 50,
    print_every: int = 100,
):

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def single_device_loss(model, batch, key):
        result = jax.vmap(model.forward_pair)(key, batch)
        return jnp.mean((result[0]-result[1])**2)

    @eqx.filter_jit
    def train_step(
        model: Data2VecVision,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
        opt_update
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(model, batch, keys)
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        model = eqx.tree_at(lambda x: x.ema.model, model, model.ema.step(model.encoder))
        return model, opt_state, loss_values

    def train_epoch(
        model: Data2VecVision,
        opt_state: PyTree,
        trainloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
    ):
        train_sampler.set_epoch(epoch)
        train_loss = 0
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch[0])
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,28,28)

            arrays = jax.device_put(jnp.split(local_batch, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            model, opt_state, loss_values = train_step(model, opt_state, global_batch, subkey, optimizer.update)
            if log_loss: train_loss += jnp.sum(process_allgather(loss_values))
            train_loss = train_loss/ jax.process_count()
            return model, opt_state, train_loss

    @eqx.filter_jit
    def test_step(
        model: Data2VecVision,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values = single_device_loss(model, batch, keys)
        return loss_values

    def test_epoch(
        model: Data2VecVision,
        testloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
    ):
        test_loss = 0
        test_sampler.set_epoch(epoch)
        for batch in testloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch[0])
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,28,28)

            arrays = jax.device_put(jnp.split(local_batch, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            test_loss += jnp.sum(process_allgather(test_step(model, global_batch, subkey)))
        test_loss_values = test_loss/ jax.process_count()
        return test_loss_values    


    
    devices = np.array(jax.devices())
    global_mesh = jax.sharding.Mesh(devices, ('b'))
    sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec(('b'),))

    max_loss = 1e10
    best_model = model
    for step in range(steps):
        if step % print_every != 0:
            key, subkey = jax.random.split(key)
            model, opt_state, train_loss = train_epoch(model, opt_state, trainloader, subkey, step, log_loss=False)
        if step % print_every == 0:
            key, subkey = jax.random.split(key)
            model, opt_state, train_loss = train_epoch(model, opt_state, trainloader, subkey, step, log_loss=True)
            key, subkey = jax.random.split(key)
            test_loss = test_epoch(model, testloader, subkey, step)

            if max_loss > test_loss:
                max_loss = test_loss
                best_model = model
            if jax.process_index() == 0:
                print(f"Step: {step}, Train Loss: {train_loss}, Test Loss: {test_loss}")
                
            
    return best_model, opt_state

model, opt_state= train(model, trainloader, testloader, jax.random.PRNGKey(0), steps=20, print_every=4)
model.save_model("best_model")