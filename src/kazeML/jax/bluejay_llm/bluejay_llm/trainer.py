from bluejay_llm.bluejay import GPT
from bluejay_llm.dataloader import ThePileDataset

import jax
import jax.numpy as jnp
import optax
import equinox
import numpy as np
import wandb
import matplotlib.pyplot as plt
import fire

import torch

class BlueJayTrainer:

    test: int

    def __init__(self) -> None:
        pass

    def train(self):
        print(1)

if __name__ == "__main__":
    fire.Fire(BlueJayTrainer)