from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from sde_score import ScordBasedSDE, GaussianFourierFeatures, LangevinCorrector
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
from sde import VESDE
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
STEPS = 50
PRINT_EVERY = 4
SEED = 5678
NUM_WORKERS = 4
TIME_FEATURE = 128
AUTOENCODER_EMBED_DIM = 256

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

unet = Unet(2, [1,16,32,64,128], AUTOENCODER_EMBED_DIM, key, group_norm_size = 32)
time_embed = eqx.nn.Sequential([
    eqx.nn.Linear(TIME_FEATURE, AUTOENCODER_EMBED_DIM, key=jax.random.PRNGKey(57104)),
    eqx.nn.Lambda(lambda x: jax.nn.swish(x))])
sde_func = VESDE(sigma_min=0.3,sigma_max=10,N=1000) # Choosing the sigma drastically affects the training speed
model = ScordBasedSDE(unet,
                    GaussianFourierFeatures(128, subkey),
                    time_embed,
                    lambda x: 1,
                    sde_func,
                    corrector=LangevinCorrector(sde_func, lambda x: x, 0.017, 1))

optimizer = optax.adam(LEARNING_RATE)

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "./data/MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "./data/MNIST",
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

key = jax.random.PRNGKey(9527)

print(jax.vmap(model.loss)(jnp.array(next(iter(trainloader))[0]),jax.random.split(jax.random.PRNGKey(100),256)).mean())
mask = np.ones((1,28,28))
mask[:,10:18,10:18] = 0
key, subkey = jax.random.split(key)
images = model.sample(subkey, (1,28,28) , 100)
key, subkey = jax.random.split(key)
inpainted_image = model.inpaint(subkey, jnp.array(next(iter(trainloader))[0][0]), mask, 300)
model, opt_state = train(model, trainloader, testloader, key, steps = STEPS, print_every=PRINT_EVERY)

key, subkey = jax.random.split(key)
images = jax.vmap(model.sample, in_axes=(0, None, None))(jax.random.split(subkey,16), (1,28,28) , 1000)
sample_grid = make_grid(torch.tensor(np.asarray(images[2])), nrow=int(np.sqrt(4)))
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.savefig("./generate")

key, subkey = jax.random.split(key)
mask[:,12:16,12:16] = 0
inpainted_image = jax.vmap(model.inpaint, in_axes=(0, 0, None, None))(jax.random.split(subkey,16), jnp.array(next(iter(trainloader))[0][:16]), mask, 1000)
sample_grid = make_grid(torch.tensor(np.asarray(inpainted_image[1])), nrow=int(np.sqrt(4)))
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.savefig("./inpaint")