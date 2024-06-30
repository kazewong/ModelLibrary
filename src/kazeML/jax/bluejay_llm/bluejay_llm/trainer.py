from bluejay_llm.bluejay import GPT
from bluejay_llm.dataloader import ThePileDataset

from jaxtyping import PRNGKeyArray, PyTree, Float, Array, Int
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tap import Tap
from jax.experimental.multihost_utils import process_allgather

import torch
from torch.utils.data import DataLoader, random_split, BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from typing import Literal


class BlueJayExperimentParser(Tap):

    # Metadata about the experiment
    data_path: str
    experiment_name: str
    project_name: str = "bluejay"
    distributed: bool = False
    mode: Literal["train", "eval"] = "train"

    # Training hyperparameters
    n_epochs: int = 10
    batch_size: int = 8
    start_learning_rate: float = 6e-4
    end_learning_rate: float = 6e-5
    warmup_steps: int = 2000
    decay_steps: int = 600000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    seed: int = 1029741092480
    num_workers: int = 8
    prefetch_factor: int = 2

    # Logging hyperparameters
    logging: bool = True
    log_epoch: int = 2
    log_t_step: int = 10
    output_path: str = "./experiment"


class BlueJayModelParser(Tap):
    # Model hyperparameters
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class BigParser(BlueJayExperimentParser, BlueJayModelParser):
    pass


class BlueJayTrainer:

    def __init__(
        self, train_set: ThePileDataset, test_set: ThePileDataset, config: BigParser
    ) -> None:
        self.config = config
        # if self.config.logging and jax.process_index() == 0:
        #     wandb.init(
        #         project=self.config.project_name,
        #         name=self.config.experiment_name,
        #         config=config.as_dict(),
        #     )

        n_processes = jax.process_count()
        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ("batch"))
        self.sharding = jax.sharding.NamedSharding(
            self.global_mesh,
            jax.sharding.PartitionSpec(
                ("batch"),
            ),
        )

        print("Creating dataloaders")

        if config.distributed:
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
        else:
            train_sampler = BatchSampler(
                SequentialSampler(train_set),
                batch_size=config.batch_size,
                drop_last=True,
            )
            test_sampler = BatchSampler(
                SequentialSampler(test_set),
                batch_size=config.batch_size,
                drop_last=True,
            )
        self.train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=train_sampler,
            prefetch_factor=config.prefetch_factor,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=test_sampler,
            prefetch_factor=config.prefetch_factor,
            pin_memory=False,
        )

        print("Creating model and optimizer")

        self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))

        self.model = GPT(
            config.vocab_size,
            config.block_size,
            config.n_layer,
            config.n_embd,
            config.dropout,
            config.bias,
            key=subkey,
        )
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=config.start_learning_rate,
            peak_value=config.start_learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_value=config.end_learning_rate,
        )
        self.optimizer = optax.chain(
            optax.adamw(
                scheduler,
                b1=config.beta1,
                b2=config.beta2,
                weight_decay=config.weight_decay,
            ),
            optax.clip_by_global_norm(1.0),
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        if jax.process_index() == 0:
            print("Start training")

        for epoch in range(self.config.n_epochs):
            print(f"Starting epoch {epoch}")
            self.model, self.opt_state, loss = self.run_epoch(
                self.model,
                self.opt_state,
                self.train_loader,
                self.key,
                epoch,
                log_loss=True,
                train=True,
            )

            if self.config.logging and jax.process_index() == 0:
                wandb.log({"train_loss": loss})

            if epoch % self.config.log_epoch == 0:
                self.model, _, loss = self.run_epoch(
                    self.model,
                    self.opt_state,
                    self.test_loader,
                    self.key,
                    epoch,
                    log_loss=True,
                    train=False,
                )

                if self.config.logging and jax.process_index() == 0:
                    wandb.log({"test_loss": loss})

    @staticmethod
    @eqx.filter_jit
    def run_step(
        model: GPT,
        opt_state: PyTree,
        input: Float[Array, "batch n_seq"],
        target: Int[Array, "batch n_seq"],
        key: PRNGKeyArray,
        opt_update,
        train: bool = True,
    ) -> tuple[GPT, PyTree, Array | float]:
        keys = jax.random.split(key, input.shape[0])
        single_device_loss = lambda model, input, target, keys: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                jax.vmap(model)(input, keys), target
            )
        )
        if train:
            loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(
                model, input, target, keys
            )
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_values
        else:
            loss_values = single_device_loss(model, input, target, keys)
            return model, opt_state, loss_values

    def run_epoch(
        self,
        model: GPT,
        opt_state: PyTree,
        data_loader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
        train: bool = True,
    ) -> tuple[GPT, PyTree, Array | float]:
        data_loader.sampler.set_epoch(epoch)  # type: ignore
        loss = 0.0
        print("Starting epoch")
        if self.config.distributed:
            for batch in data_loader:
                key, subkey = jax.random.split(key)

                local_input = jnp.array(batch[0])
                local_target = jnp.array(batch[1])

                global_shape = (
                    jax.process_count() * local_input.shape[0],
                ) + local_input.shape[1:]

                batch_input = jax.device_put(
                    jnp.split(local_input, len(self.global_mesh.local_devices), axis=0),
                    self.global_mesh.local_devices,
                )
                batch_target = jax.device_put(
                    jnp.split(
                        local_target, len(self.global_mesh.local_devices), axis=0
                    ),
                    self.global_mesh.local_devices,
                )

                global_input = jax.make_array_from_single_device_arrays(
                    global_shape, self.sharding, batch_input
                )
                global_target = jax.make_array_from_single_device_arrays(
                    global_shape, self.sharding, batch_target
                )

                model, opt_state, loss_values = self.run_step(
                    model,
                    opt_state,
                    global_input,
                    global_target,
                    subkey,
                    self.optimizer.update,
                    train=train,
                )
        else:
            for batch in data_loader:
                key, subkey = jax.random.split(key)
                input = jnp.array(batch[0])
                target = jnp.array(batch[1])
                model, opt_state, loss_values = self.run_step(
                    model,
                    opt_state,
                    input,
                    target,
                    subkey,
                    self.optimizer.update,
                    train=train,
                )
                if log_loss:
                    loss += loss_values
        return model, opt_state, loss
