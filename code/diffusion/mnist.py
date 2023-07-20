from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from sde_score import ScordBasedSDE, GaussianFourierFeatures
from common.Unet import Unet
import jax
import jax.numpy as jnp
import torchvision
import torch
import tqdm
import optax
import equinox as eqx
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 100
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 4

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

unet = Unet(2, [1,16,32,64,128], 256, key, group_norm_size = 32)
sde = ScordBasedSDE(unet, lambda x: 1.0, lambda x: 1.0, lambda x: 25**x, lambda x: jnp.sqrt((25**(2 * x) - 1.) / 2. / jnp.log(25)), GaussianFourierFeatures(128, subkey))
optimizer = optax.adam(LEARNING_RATE)

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
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

def train(
    sde: ScordBasedSDE,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
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
        batch_loss = lambda model, batch, key: jnp.mean(jax.vmap(model.loss)(batch, key))
        loss_values, grads = eqx.filter_value_and_grad(batch_loss)(model, batch, keys)
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values
    
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices,)+tuple(jnp.ones(sde.n_dim+1,dtype=int).tolist()))
    shard = sharding.PositionalSharding(devices)

    max_loss = 1e10
    for step in tqdm.trange(steps):
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            batch = jnp.array(batch[0])
            # batch = jax.device_put(batch, shard)
            sde, opt_state, loss_values = make_step(sde, opt_state, batch, subkey, optimizer.update)
        if step % print_every == 0:
            test_loss = 0
            for batch in testloader:
                key, subkey = jax.random.split(key)
                batch = jnp.array(batch[0])
                subkey = jax.random.split(subkey,batch.shape[0])
                test_loss += jnp.mean(jax.vmap(sde.loss)(batch, subkey))
            test_loss_values = test_loss / len(testloader)
            if max_loss > test_loss_values:
                max_loss = test_loss_values
                best_model = sde
                print(f"test loss: {test_loss_values}")
            print(f"Step {step}: {loss_values}")



    return best_model, opt_state

sde, opt_state = train(sde, trainloader, testloader, key, steps = STEPS, print_every=PRINT_EVERY)
key = jax.random.PRNGKey(9527)
images = sde.sample((1,28,28) ,subkey, 300, 4)