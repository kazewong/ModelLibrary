from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from sde_score import ScordBasedSDE, GaussianFourierFeatures
from common.Unet import Unet
import jax
import jax.numpy as jnp
import torchvision
import tqdm
import optax
import equinox as eqx
from jax._src.distributed import initialize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import mlflow

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 200
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 4
TIME_FEATURE = 128
AUTOENCODER_EMBED_DIM = 256

mlflow.log_params({
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "steps": STEPS,
    "seed": SEED,
    "num_workers": NUM_WORKERS,
    "time_feature": TIME_FEATURE,
    "autoencoder_embed_dim": AUTOENCODER_EMBED_DIM
})
initialize()
print(jax.process_count())
print(jax.devices())
print(jax.local_device_count())

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

unet = Unet(2, [1,16,32,64,128], AUTOENCODER_EMBED_DIM, key, group_norm_size = 32)
time_embed = eqx.nn.Linear(TIME_FEATURE, AUTOENCODER_EMBED_DIM, key=jax.random.PRNGKey(57104))
sde = ScordBasedSDE(unet,
                    lambda x: 1,
                    lambda x: 1.0,
                    lambda x: 25**x,
                    lambda x: jnp.sqrt((25**(2 * x) - 1.) / 2. / jnp.log(25)),
                    GaussianFourierFeatures(128, subkey),
                    time_embed)

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

print("Creating sampler")

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

print("Creating dataloader")

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
    sde: ScordBasedSDE,
    trainloader: DataLoader,
    testloader: DataLoader,
    key: PRNGKeyArray,
    steps: int = 1000,
    print_every: int = 100,
):

    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: ScordBasedSDE,
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
    
    devices = np.array(jax.devices())
    global_mesh = jax.sharding.Mesh(devices, ('b'))
    sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec(('b'),))

    max_loss = 1e10
    loss_values = 0
    best_model = sde
    for step in tqdm.trange(steps):
        train_sampler.set_epoch(step)
        test_sampler.set_epoch(step)
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch[0])
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,28,28)

            arrays = jax.device_put(jnp.split(local_batch, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            sde, opt_state, loss_values = make_step(sde, opt_state, global_batch, subkey, optimizer.update)
        
        if jax.process_index() == 0:
            if step % print_every == 0:
                mlflow.log_metric(key="training_loss", value=loss_values, step=step)

                # test_loss = 0
                # for batch in testloader:
                #     key, subkey = jax.random.split(key)
                #     batch = jnp.array(batch[0])
                #     subkey = jax.random.split(subkey,batch.shape[0])
                #     test_loss += jnp.mean(jax.vmap(sde.loss)(batch, subkey))
                # test_loss_values = test_loss / len(testloader)
                # if max_loss > test_loss_values:
                #     max_loss = test_loss_values
                #     best_model = sde
                #     print(f"test loss: {test_loss_values}")
                print(f"Step {step}: {loss_values}")

    return best_model, opt_state

sde, opt_state = train(sde, trainloader, testloader, key, steps = STEPS, print_every=PRINT_EVERY)