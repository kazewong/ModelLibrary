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
import numpy as np
from kazeML.jax.data2vec.data2vec_dataset import Data2VecDataset
from kazeML.jax.common.Transformer import TransformerConfig


class Data2VecTrainer(Tap):
    # Metadata about the experiment
    data_path: str
    experiment_name: str
    project_name: str = "Data2Vec"
    distributed: bool = False

    # Transformer Config
    transformer_config: TransformerConfig

    # Model hyperparameters
    time_feature: int = 128
    autoencoder_embed_dim: int = 256
    hidden_layer: list[int] = [3, 16, 32, 64, 128]
    group_norm_size: int = 32

    # Training hyperparameters
    n_epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    log_epoch: int = 2
    seed: int = 2019612721831
    num_workers: int = 8
    train_test_ratio: float = 0.8


class Data2VecTrainer:
    def __init__(self, config: Data2VecTrainer, logging: bool = False):
        self.config = config
        self.logging = logging
        if logging and (jax.process_index() == 0):
            Task.init(project_name=args.project_name, task_name=args.experiment_name)

        # Initialize distributed training
        n_processes = jax.process_count()
        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ("b"))
        self.sharding = jax.sharding.NamedSharding(
            self.global_mesh,
            jax.sharding.PartitionSpec(
                ("b"),
            ),
        )

        # Initialize the dataset
        dataset = Data2VecDataset(config.data_path)
        train_set, test_set = random_split(
            dataset, [config.train_test_ratio, 1 - config.train_test_ratio]
        )
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=n_processes,
            rank=jax.process_index(),
            shuffle=True,
            seed=config.seed,
        )
        test_sampler = DistributedSampler(
            test_set,
            num_replicas=n_processes,
            rank=jax.process_index(),
            shuffle=False,
            seed=config.seed,
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=test_sampler,
            pin_memory=True,
        )

        self.data_shape = train_set.dataset.get_shape()

        # Initialize the model

        # Initialize the optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        if jax.process_index() == 0:
            print("Start training")
        max_loss = 1e10
        self.best_model = self.model
        for step in range(self.config.n_epochs):
            if jax.process_index() == 0:
                print("Epoch: ", step)
            if step % self.config.log_epoch == 0:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.train_epoch(
                    self.model,
                    self.opt_state,
                    self.train_loader,
                    subkey,
                    step,
                    log_loss=True,
                )
                self.key, subkey = jax.random.split(self.key)
                test_loss = self.test_epoch(self.model, self.test_loader, subkey, step)

                if max_loss > test_loss:
                    max_loss = test_loss
                    self.best_model = self.model
                if self.logging:
                    Logger.current_logger().report_scalar(
                        "Loss", "training_loss", value=train_loss, iteration=step
                    )
                    Logger.current_logger().report_scalar(
                        "Loss", "test_loss", value=test_loss, iteration=step
                    )
                    self.best_model.save_model("./best_model")
                    Task.current_task().upload_artifact(
                        artifact_object="./best_model", name="model"
                    )
            else:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.train_epoch(
                    self.model,
                    self.opt_state,
                    self.train_loader,
                    subkey,
                    step,
                    log_loss=False,
                )

    def validate(self):
        pass

    @staticmethod
    @eqx.filter_jit
    def train_step(
        model: Data2Vec,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
        opt_update,
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(
            jax.vmap(model.loss)(batch, key)
        )
        loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(
            model, batch, keys
        )
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values

    @staticmethod
    @eqx.filter_jit
    def test_step(
        model: ScoreBasedSDE,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values = jnp.mean(jax.vmap(model.loss)(batch, keys))
        return loss_values

    def train_epoch(
        self,
        model: ScoreBasedSDE,
        opt_state: PyTree,
        trainloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
    ) -> tuple[ScoreBasedSDE, PyTree, Array | float]:
        self.train_loader.sampler.set_epoch(epoch)
        train_loss = 0
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch)
            global_shape = (
                jax.process_count() * local_batch.shape[0],
            ) + self.data_shape

            arrays = jax.device_put(
                jnp.split(local_batch, len(self.global_mesh.local_devices), axis=0),
                self.global_mesh.local_devices,
            )
            global_batch = jax.make_array_from_single_device_arrays(
                global_shape, self.sharding, arrays
            )
            model, opt_state, loss_values = self.train_step(
                model, opt_state, global_batch, subkey, self.optimizer.update
            )
            if log_loss:
                train_loss += jnp.sum(process_allgather(loss_values))
        train_loss = (
            train_loss
            / jax.process_count()
            / len(trainloader)
            / np.sum(self.data_shape)
        )
        return model, opt_state, train_loss

    def test_epoch(
        self,
        model: ScoreBasedSDE,
        testloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
    ):
        test_loss = 0
        self.test_loader.sampler.set_epoch(epoch)
        for batch in testloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch)
            global_shape = (
                jax.process_count() * local_batch.shape[0],
            ) + self.data_shape

            arrays = jax.device_put(
                jnp.split(local_batch, len(self.global_mesh.local_devices), axis=0),
                self.global_mesh.local_devices,
            )
            global_batch = jax.make_array_from_single_device_arrays(
                global_shape, self.sharding, arrays
            )
            test_loss += jnp.sum(
                process_allgather(self.test_step(model, global_batch, subkey))
            )
        test_loss_values = (
            test_loss / jax.process_count() / len(testloader) / np.sum(self.data_shape)
        )
        return test_loss_values


if __name__ == "__main__":
    args = Data2VecTrainer().parse_args()

    # if args.distributed == True:
    #     initialize()
    #     print(jax.process_count())

    # n_processes = jax.process_count()
    # if jax.process_index() == 0:
    #     trainer = SDEDiffusionTrainer(args, logging=True)
    #     trainer.train()
    # else:
    #     trainer = SDEDiffusionTrainer(args, logging=False)
    #     trainer.train()
