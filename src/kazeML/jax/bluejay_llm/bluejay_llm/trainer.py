from bluejay_llm.bluejay import GPT
from bluejay_llm.dataloader import ThePileDataset

import jax
import jax.numpy as jnp
import optax
import equinox
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tap import Tap

import torch

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

    def __init__(self,

                 ) -> None:
        self.test = 1

    def train(self):
        print(self.test)
