
import jax.numpy as jnp
import torchvision
import torch
import optax
import equinox as eqx
import jax

from kazeML.jax.common.Transformer import TransformerConfig
from kazeML.jax.data2vec.data2vec_vision import Data2VecVision, Data2VecVisionConfig
from kazeML.jax.common.modules.EMA import EMAModule

transformer_config = TransformerConfig(eqx.nn.Lambda(jax.nn.gelu), embed_dim=64)
D2V_cfg = Data2VecVisionConfig(transformer_config, image_size=28, patch_size=4, in_channels=1,
                               embed_dim=64)

LEARNING_RATE = 1e-4

model = Data2VecVision(jax.random.PRNGKey(0),D2V_cfg)

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

model.forward_pair(jax.random.PRNGKey(0),jnp.array(train_dataset[0][0]))
# trainloader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
# )
# testloader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
# )