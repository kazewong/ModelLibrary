from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from sde_score import ScordBasedSDE
from common.Unet import Unet
import jax
import jax.numpy as jnp
import torchvision
import torch
import tqdm
import optax
import equinox as eqx

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
STEPS = 5
PRINT_EVERY = 30
SEED = 5678
NUM_WORKERS = 4

key = jax.random.PRNGKey(SEED)

unet = Unet(2, [1,16,32,64,128], key)
sde = ScordBasedSDE(unet, lambda x: 1.0, lambda x: 1.0, lambda x: (25**(2 * x) - 1.) / 2. / jnp.log(25))
optimizer = optax.adamw(LEARNING_RATE)

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
    
    for step in tqdm.trange(steps):
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            batch = jnp.array(batch[0])
            sde, opt_state, loss_values = make_step(sde, opt_state, batch, subkey, optimizer.update)
            print(loss_values)
        # if step % print_every == 0:
        #     print(f"Step {step}: {loss_values}")

train(sde, trainloader, testloader, key)