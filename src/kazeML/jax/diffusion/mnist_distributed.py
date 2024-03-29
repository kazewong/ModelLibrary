from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import torchvision
import optax
import equinox as eqx
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from kazeML.jax.diffusion.sde import VESDE
from kazeML.jax.diffusion.sde_score import ScoreBasedSDE, GaussianFourierFeatures
from kazeML.jax.common.Unet import Unet


BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 200
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 4
TIME_FEATURE = 128
AUTOENCODER_EMBED_DIM = 256

initialize()

if jax.process_index() == 0:
    print(jax.process_count())
    print(jax.devices())
    print(jax.local_device_count())


key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

unet = Unet(2, [1,16,32,64,128], AUTOENCODER_EMBED_DIM, key, group_norm_size = 32)
time_embed = eqx.nn.Linear(TIME_FEATURE, AUTOENCODER_EMBED_DIM, key=jax.random.PRNGKey(57104))
sde_func = VESDE(sigma_min=0.3,sigma_max=10,N=300) # Choosing the sigma drastically affects the training speed
model = ScoreBasedSDE(unet,
                    GaussianFourierFeatures(128, subkey),
                    time_embed,
                    lambda x: 1,
                    sde_func,)

optimizer = optax.adam(LEARNING_RATE)

normalize_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "./MNIST",
    train=True,
    download=True,
    transform=normalize_data,
)
test_dataset = torchvision.datasets.MNIST(
    "./MNIST",
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


def train(
    model: ScoreBasedSDE,
    trainloader: DataLoader,
    testloader: DataLoader,
    key: PRNGKeyArray,
    steps: int = 1000,
    print_every: int = 100,
):

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: ScoreBasedSDE,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
        opt_update
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(jax.vmap(model.loss)(batch, key))
        loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(model, batch, keys)
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values

    def train_epoch(
        model: ScoreBasedSDE,
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
        model: ScoreBasedSDE,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values = jnp.mean(jax.vmap(model.loss)(batch, keys))
        return loss_values

    def test_epoch(
        model: ScoreBasedSDE,
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

model, opt_state = train(model, trainloader, testloader, key, steps = STEPS, print_every=PRINT_EVERY)
if jax.process_index() == 0:
    images = model.sample((1,28,28) ,subkey)