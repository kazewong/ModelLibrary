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
from kazeML.jax.diffusion.sde_score import ScordBasedSDE, GaussianFourierFeatures, LangevinCorrector, SDEDiffusionModelParser
from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset
import numpy as np

class SDEDiffusionExperimentParser(Tap):
    # Metadata about the experiment
    mode: Literal["train", "predict"] = "train"
    data_path: str
    experiment_name: str
    project_name: str = "DiffusionAstro"
    distributed: bool = False
    conditional: bool = False


    # Training hyperparameters
    n_epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    log_epoch: int = 2
    seed: int = 2019612721831
    num_workers: int = 8
    train_test_ratio: float = 0.8

    def configure(self) -> None:
        self.add_subparser("SDEDiffusionModelParser", SDEDiffusionModelParser, help="Model hyperparameters")

class SDEDiffusionTrainer:

    def __init__(self,
                config: SDEDiffusionExperimentParser, logging: bool = False):
        self.config = config
        self.logging = logging
        if logging and (jax.process_index() == 0):
            Task.init(project_name=args.project_name, task_name=args.experiment_name)

        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ('b'))
        self.sharding = jax.sharding.NamedSharding(self.global_mesh, jax.sharding.PartitionSpec(('b'),))

        train_set, test_set = random_split(DiffusionDataset(self.config.data_path), [config.train_test_ratio, 1 - config.train_test_ratio])
        train_sampler = DistributedSampler(train_set,
                                           num_replicas=n_processes,
                                           rank=jax.process_index(),
                                           shuffle=True,
                                           seed=config.seed)
        test_sampler = DistributedSampler(test_set,
                                            num_replicas=n_processes,
                                            rank=jax.process_index(),
                                            shuffle=False,
                                            seed=config.seed)
        self.train_loader = DataLoader(train_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        sampler=train_sampler,
                                        pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        sampler=test_sampler,
                                        pin_memory=True)

        self.data_shape = train_set.dataset.get_shape()


        self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
        unet = Unet(len(self.data_shape)-1, config.hidden_layer, config.autoencoder_embed_dim, subkey, group_norm_size=config.group_norm_size)
        self.key, subkey = jax.random.split(self.key)
        time_embed = eqx.nn.Linear(config.time_feature, config.autoencoder_embed_dim, key=subkey)
        self.key, subkey = jax.random.split(self.key)
        gaussian_feature = GaussianFourierFeatures(config.time_feature, subkey)
        sde_func = VESDE(sigma_min=0.3,sigma_max=10,N=1000) # Choosing the sigma drastically affects the training speed
        self.model = ScordBasedSDE(unet,
                                    gaussian_feature,
                                    time_embed,
                                    lambda x: 1,
                                    sde_func,
                                    corrector=LangevinCorrector(sde_func, lambda x: x, 0.017, 1),)

        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        if jax.process_index()==0: print("Start training")
        max_loss = 1e10
        self.best_model = self.model
        for step in range(self.config.n_epochs):
            if jax.process_index()==0: print("Epoch: ", step)
            if step % self.config.log_epoch == 0:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.run_epoch(self.model, self.opt_state, self.train_loader, subkey, step, log_loss=True, train=True)
                self.key, subkey = jax.random.split(self.key)
                _, _, test_loss = self.run_epoch(self.model, self.opt_state, self.test_loader, subkey, step, log_loss=True, train=False)

                if max_loss > test_loss:
                    max_loss = test_loss
                    self.best_model = self.model
                if self.logging:
                    Logger.current_logger().report_scalar("Loss", "training_loss", value=train_loss, iteration=step) # type: ignore
                    Logger.current_logger().report_scalar("Loss", "test_loss", value=test_loss, iteration=step) # type: ignore
                    self.best_model.save_model("./best_model")
                    Task.current_task().upload_artifact(artifact_object="./best_model", name="model")
            else:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.run_epoch(self.model, self.opt_state, self.train_loader, subkey, step, log_loss=False, train=True)


    def validate(self):
        pass

    @staticmethod
    @eqx.filter_jit
    def run_step(
        model: ScordBasedSDE,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
        opt_update,
        train: bool = True,
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(jax.vmap(model.loss)(batch, key))
        if train:
            loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(model, batch, keys)
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_values
        else:
            loss_values = single_device_loss(model, batch, keys)
            return model, opt_state, loss_values

    def run_epoch(self,
        model: ScordBasedSDE,
        opt_state: PyTree,
        data_loader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
        train: bool = True,
    ) -> tuple[ScordBasedSDE, PyTree, Array | float]:
        data_loader.sampler.set_epoch(epoch)
        loss = 0
        for batch in data_loader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch)
            global_shape = (jax.process_count() * local_batch.shape[0], ) + self.data_shape

            arrays = jax.device_put(jnp.split(local_batch, len(self.global_mesh.local_devices), axis = 0), self.global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, self.sharding, arrays)
            model, opt_state, loss_values = self.run_step(model, opt_state, global_batch, subkey, self.optimizer.update, train=train)
            if log_loss: loss += jnp.sum(process_allgather(loss_values))
        loss = loss/ jax.process_count() / len(data_loader) /np.sum(self.data_shape)
        return model, opt_state, loss
            

if __name__ == "__main__":

    args = SDEDiffusionExperimentParser().parse_args()

    if args.distributed == True:
        initialize()
        print(jax.process_count())

    n_processes = jax.process_count()
    if jax.process_index() == 0:
        trainer = SDEDiffusionTrainer(args, logging=True)
        trainer.train()
    else:
        trainer = SDEDiffusionTrainer(args, logging=False)
        trainer.train()
