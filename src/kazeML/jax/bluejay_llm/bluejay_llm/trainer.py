from bluejay_llm.bluejay import GPT
from bluejay_llm.dataloader import ThePileDataset

from jaxtyping import PRNGKeyArray, PyTree, Float, Array
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tap import Tap

import torch
from torch.utils.data import DataLoader, random_split
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
    learning_rate: float = 3e-4
    seed: int = 1029741092480
    num_workers: int = 8
    train_test_ratio: float = 0.8
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

    def __init__(self, dataset: ThePileDataset, config: BigParser) -> None:
        self.config = config
        if self.config.logging and jax.process_index() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=config.as_dict(),
            )

        n_processes = jax.process_count()
        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ("batch"))
        self.sharding = jax.sharding.NamedSharding(
            self.global_mesh,
            jax.sharding.PartitionSpec(
                ("batch"),
            ),
        )

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
        learning_rate = config.learning_rate
        self.optimizer = optax.chain(
            optax.adamw(learning_rate), optax.clip_by_global_norm(1.0)
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        pass

    @staticmethod
    def run_step(
        model: GPT,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
        opt_update,
        train: bool = True,
    ):
        pass

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
        return model, opt_state, 0.0
