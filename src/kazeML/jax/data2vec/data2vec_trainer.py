import json
import os
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
from kazeML.jax.data2vec.data2vec import Data2Vec, Data2VecConfig
from kazeML.jax.data2vec.data2vec_dataset import Data2VecDataset
from kazeML.jax.common.Transformer import TransformerConfig
from kazeML.jax.data2vec.feature_extractor import SeriesFeatureExtractor


class Data2VecTrainerParser(Tap):
    # Metadata about the experiment
    data_path: str
    experiment_name: str
    project_name: str = "Data2Vec"
    distributed: bool = False

    # FeatureExtractor hyperparameters
    FE_channels: list[int] = [1, 8, 16, 32]
    FE_kernels: list[int] = [3, 3, 3, 3]
    FE_strides: list[int] = [10, 5, 2, 2]
    FE_dropout: float = 0.0
    FE_affine_group_norm: bool = False
    FE_log_compression: bool = False
    FE_skip_connections: bool = False
    FE_residual_scale: float = 1.0

    # Transformer hyperparameters

    max_length: int = 512
    embed_dim: int = 512
    layernorm_embedding: bool = False
    ffn_embed_dim: int = 2048
    layers: int = 6
    attention_heads: int = 8
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0

    # Data2Vec hyperparameters

    n_example: int = 5
    mask_fraction: float = 0.1
    mask_length: float = 0.1
    min_masks: int = 1

    # Training hyperparameters
    n_epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    seed: int = 2019612721831
    num_workers: int = 8
    train_test_ratio: float = 0.8

    # Logging hyperparameters
    log_epoch: int = 2
    log_t_step: int = 10
    output_path: str = "./experiment"


class Data2VecTrainer:
    def __init__(self, dataset: Data2VecDataset, config: Data2VecTrainerParser, logging: bool = False):
        self.config = config
        self.logging = logging
        if logging and (jax.process_index() == 0):
            Task.init(project_name=config.project_name, task_name=config.experiment_name)

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
        dataset = Data2VecDataset(config.data_path,
        n_example=config.n_example,
        mask_fraction=config.mask_fraction,
        mask_length=config.mask_length,
        min_masks=config.min_masks,
        seed=config.seed+4,)

        key = jax.random.PRNGKey(config.seed + 3)
        key, subkey = jax.random.split(key)
        layer_spec = list(
            zip(
                config.FE_channels,
                config.FE_kernels,
                config.FE_strides,
            )
        )
        if dataset.n_dim == 1:
            feature_extractor = SeriesFeatureExtractor(
                subkey,
                layer_spec,
                p_dropout=config.FE_dropout,
                affine_group_norm=config.FE_affine_group_norm,
                log_compression=config.FE_log_compression,
                skip_connections=config.FE_skip_connections,
                residual_scale=config.FE_residual_scale,
            )
        else:
            raise NotImplementedError

        dataset.set_data_length(feature_extractor)

        self.data_shape = dataset.data_shape
        self.mask_shape = (config.n_example, dataset.data_length)

        config_dict = config.as_dict()
        config_dict['max_length'] = dataset.data_length
        config_dict['embed_dim'] = config_dict['FE_channels'][-1]

        transformer_config = TransformerConfig(
            activation=jax.nn.gelu,
            **config_dict
        )

        data2Vec_config = Data2VecConfig(transformer_encoder_config=transformer_config, **config_dict)
        self.model = Data2Vec(
            subkey, feature_extractor=feature_extractor, config=data2Vec_config
        )

        train_set, test_set = random_split(
            dataset, [config.train_test_ratio, 1 - config.train_test_ratio]
        )
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=n_processes,
            rank=jax.process_index(),
            shuffle=True,
            seed=config.seed + 1,
        )
        test_sampler = DistributedSampler(
            test_set,
            num_replicas=n_processes,
            rank=jax.process_index(),
            shuffle=False,
            seed=config.seed + 2,
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

        # Initialize the optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.key = jax.random.PRNGKey(config.seed+5)

    def train(self):
        if jax.process_index() == 0:
            print("Start training")
        max_loss = 1e10
        self.best_model = self.model
        logging_key, logging_subkey = jax.random.split(self.key)

        for step in range(self.config.n_epochs):
            if jax.process_index() == 0:
                print("Epoch: ", step)
            if step % self.config.log_epoch == 0:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.run_epoch(
                    self.model,
                    self.opt_state,
                    self.train_loader,
                    subkey,
                    step,
                    log_loss=True,
                    train=True
                )
                self.key, subkey = jax.random.split(self.key)
                _, _, test_loss = self.run_epoch(self.model, self.opt_state, self.test_loader, subkey, step, log_loss=True, train=False)

                if max_loss > test_loss:
                    max_loss = test_loss
                    self.best_model = self.model
                    self.best_model.save_model(self.config.output_path + "/best_model")

                if self.logging:
                    logging_key, subkey = jax.random.split(logging_key)
                    Logger.current_logger().report_scalar("Loss", "training_loss", value=float(train_loss), iteration=step)  # type: ignore
                    Logger.current_logger().report_scalar("Loss", "test_loss", value=float(test_loss), iteration=step)  
                    self.model.save_model(self.config.output_path + "/latest_model")
                    Task.current_task().upload_artifact(
                        name="latest_model",
                        artifact_object=self.config.output_path + "/latest_model.eqx",
                        metadata={"step": step},
                    )
                    Task.current_task().upload_artifact(
                        "best_model",
                        artifact_object=self.config.output_path + "/best_model.eqx",
                        metadata={"step": step},
                    )
            else:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.run_epoch(
                    self.model,
                    self.opt_state,
                    self.train_loader,
                    subkey,
                    step,
                    log_loss=False,
                )


    @staticmethod
    @eqx.filter_jit
    def run_step(
        model: Data2Vec,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        mask: Float[Array, "batch n_example datashape"],
        key: PRNGKeyArray,
        opt_update,
        train: bool = True,
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, mask, key: jnp.mean(
            jax.vmap(model.d2v_loss)(batch, mask, key)
        )
        if train:
            loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(
                model, batch, mask, keys
            )
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            model = eqx.tree_at(lambda x: x.ema.model, model, model.ema.step(model.encoder))
            return model, opt_state, loss_values
        else:
            loss_values = single_device_loss(model, batch, mask, keys)
            return model, opt_state, loss_values

    def run_epoch(
        self,
        model: Data2Vec,
        opt_state: PyTree,
        data_loader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
        train: bool = True
    ) -> tuple[Data2Vec, PyTree, Float[Array, "1"]]:
        data_loader.sampler.set_epoch(epoch) # type: ignore
        loss = jnp.array([0.])
        for batch in data_loader:
            key, subkey = jax.random.split(key)
            local_batch, local_mask = jnp.array(batch[0]), jnp.array(batch[1])
            global_batch_shape = (
                jax.process_count() * local_batch.shape[0],
            ) + self.data_shape

            global_mask_shape = (
                jax.process_count() * local_batch.shape[0],
            ) + self.mask_shape


            batch_arrays = jax.device_put(
                jnp.split(local_batch, len(self.global_mesh.local_devices), axis=0),
                self.global_mesh.local_devices,
            )

            batch_mask = jax.device_put(
                jnp.split(local_mask, len(self.global_mesh.local_devices), axis=0),
                self.global_mesh.local_devices,
            )

            global_batch = jax.make_array_from_single_device_arrays(
                global_batch_shape, self.sharding, batch_arrays
            )

            global_mask = jax.make_array_from_single_device_arrays(
                global_mask_shape, self.sharding, batch_mask
            )

            model, opt_state, loss_values = self.run_step(
                model, opt_state, global_batch, global_mask, subkey, self.optimizer.update, train=train
            )
            if log_loss:
                loss += jnp.sum(process_allgather(loss_values))
        loss = (
            loss
            / jax.process_count()
            / len(data_loader)
        )
        return model, opt_state, loss


