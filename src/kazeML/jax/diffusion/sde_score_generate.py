from typing import Literal
from tap import Tap
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from clearml import Task, Logger
from kazeML.jax.common.Unet import Unet
from kazeML.jax.diffusion.sde import VESDE
from kazeML.jax.diffusion.sde_score import ScoreBasedSDE, GaussianFourierFeatures, LangevinCorrector
from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset
from kazeML.jax.diffusion.sde_score_trainer import SDEDiffusionModelParser
import numpy as np

class SDEDiffusionPipelineParser(Tap):

    model: Literal['unconditional', 'conditional', 'inpaint'] = 'unconditional'

    # Model hyperparameters
    model_path: str
    config_path: str

    # Pipeline hyperparameters

    output_path: str
    seed: int = 0

class BigParser(SDEDiffusionPipelineParser, SDEDiffusionModelParser):
    pass

class SDEDiffusionPipeline:

        def __init__(self,
                config: SDEDiffusionPipelineParser):
            self.config = config

            # n_processes = jax.process_count()
            # devices = np.array(jax.devices())
            # self.global_mesh = jax.sharding.Mesh(devices, ('b'))
            # self.sharding = jax.sharding.NamedSharding(self.global_mesh, jax.sharding.PartitionSpec(('b'),))

            self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
            unet = Unet(len(self.data_shape)-1, config.hidden_layer, config.autoencoder_embed_dim, subkey, group_norm_size=config.group_norm_size)
            self.key, subkey = jax.random.split(self.key)
            time_embed = eqx.nn.Linear(config.time_feature, config.autoencoder_embed_dim, key=subkey)
            self.key, subkey = jax.random.split(self.key)
            gaussian_feature = GaussianFourierFeatures(config.time_feature, subkey, scale=config.scale)
            sde_func = VESDE(sigma_min=config.sigma_min, sigma_max=config.sigma_max, N=config.N)
            self.model = ScoreBasedSDE(unet,
                                        gaussian_feature,
                                        time_embed,
                                        lambda x: 1,
                                        sde_func,
                                        corrector=LangevinCorrector(sde_func, lambda x: x, 0.017, 1),)



if __name__ == "__main__":

    args = SDEDiffusionPipelineParser().parse_args()
    pipeline = SDEDiffusionPipeline(args)