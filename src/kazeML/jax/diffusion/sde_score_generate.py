from typing import Literal
from tap import Tap
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from kazeML.jax.common.Unet import Unet, UnetConfig
from kazeML.jax.diffusion.sde import VESDE
from kazeML.jax.diffusion.sde_score import (
    ScoreBasedSDE,
    GaussianFourierFeatures,
    LangevinCorrector,
)
from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset
from kazeML.jax.diffusion.sde_score_trainer import SDEDiffusionModelParser
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt


class SDEDiffusionPipelineParser(Tap):
    model: Literal["unconditional", "conditional", "inpaint"] = "unconditional"

    # Model hyperparameters
    model_path: str
    config_path: str
    data_shape: tuple = (1, 64, 64)

    # Pipeline hyperparameters

    output_path: str
    seed: int = 0


class BigParser(SDEDiffusionPipelineParser, SDEDiffusionModelParser):
    pass


class SDEDiffusionPipeline:
    def __init__(self, config: BigParser):
        with open(config.config_path, "r") as file:
            config_file_dict = json.load(file)
        config = config.from_dict(config_file_dict)

        self.config = config

        self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
        
        unet_config = UnetConfig(
            num_dim=len(config.data_shape[1:]),
            input_channels=config.data_shape[0],
            output_channels=config.data_shape[0],
            embedding_dim=config.embedding_dim,
            base_channels=config.base_channels,
            n_resolution=config.n_resolution,
            n_resnet_blocks=config.n_resnet_blocks,
            kernel_size=config.kernel_size,
            stride=config.stride,
            dilation=config.dilation,
            group_norm_size=config.group_norm_size,
            dropout=config.dropout,
            padding=config.padding,
            skip_rescale=config.skip_rescale,
            sampling_method=config.sampling_method,
            fir=config.fir,
            fir_kernel_size=config.fir_kernel_size,
        )

        unet = Unet(subkey, unet_config, activation=jax.nn.swish)
        self.key, subkey = jax.random.split(self.key)
        time_embed = eqx.nn.Linear(
            config.time_feature, config.embedding_dim, key=subkey
        )
        self.key, subkey = jax.random.split(self.key)
        gaussian_feature = GaussianFourierFeatures(
            config.time_feature, subkey, scale=config.scale
        )
        sde_func = VESDE(
            sigma_min=config.sigma_min, sigma_max=config.sigma_max, N=config.N
        )
        self.model = ScoreBasedSDE(
            unet,
            gaussian_feature,
            time_embed,
            lambda x: 1,
            sde_func,
            corrector=LangevinCorrector(sde_func, lambda x: x, 0.017, 1),
        )

        self.model = self.model.load_model(config.model_path)

    def unconditional_sampling(
        self, key: PRNGKeyArray, n_samples: int, n_steps: int
    ) -> Array:
        key = jax.random.split(key, n_samples)
        return jax.vmap(self.model.sample, in_axes=(0, None, None, None))(
            key, self.config.data_shape, n_steps, 1e-5
        )

    def inpaint(
        self, key: PRNGKeyArray, data: Array, mask: Array, n_steps: int
    ) -> Array:
        return self.model.inpaint(key, data, mask, n_steps)


if __name__ == "__main__":
    args = SDEDiffusionPipelineParser().parse_args()
    pipeline = SDEDiffusionPipeline(args)
    model = pipeline.model
    size = 256
    data = h5py.File("/mnt/home/wwong/ceph/Dataset/ThereIsASky/galaxyzoo/images_gz2.hdf5")
    image_masked = (data['data'][100][:,212-size//2:212+size//2,212-size//2:212+size//2]/ 255).astype(np.float32) - 0.5
    mask = np.ones((3, size, size), dtype=np.float32)
    mask_size = 64
    mask[:, size//2-mask_size//2:size//2+mask_size//2, size//2-mask_size//2:size//2+mask_size//2] = 0
    # mask[:, 3*size//4-mask_size//2:3*size//4+mask_size//2, size//4-mask_size//2:size//4+mask_size//2] = 0
    # mask[:, size//4-mask_size//2:size//4+mask_size//2, 3*size//4-mask_size//2:3*size//4+mask_size//2] = 0
    inpainted = pipeline.inpaint(jax.random.PRNGKey(0), image_masked, mask, 2000)
    plt.imshow(inpainted[1].T+0.5)
    plt.savefig("test.png")