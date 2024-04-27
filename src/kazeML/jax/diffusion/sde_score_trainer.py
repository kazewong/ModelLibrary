from dataclasses import asdict
from typing import Callable, Literal
from tap import Tap
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import copy
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from kazeML.jax.common.Unet import Unet, UnetConfig
from kazeML.jax.diffusion.sde import VESDE
from kazeML.jax.diffusion.sde_score import (
    ScoreBasedSDE,
    GaussianFourierFeatures,
    LangevinCorrector,
)
from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset

import wandb
import matplotlib.pyplot as plt


class SDEDiffusionExperimentParser(Tap):
    # Metadata about the experiment
    mode: Literal["train", "predict"] = "train"
    data_path: str
    experiment_name: str
    project_name: str = "DiffusionAstro"
    distributed: bool = False
    conditional: bool = False
    debug: bool = False

    # Training hyperparameters
    n_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    seed: int = 2019612721831
    num_workers: int = 8
    train_test_ratio: float = 0.8
    prefetch_factor: int = 2

    # Logging hyperparameters
    log_epoch: int = 2
    log_t_step: int = 10
    output_path: str = "./experiment"



class SDEDiffusionModelParser(Tap):
    SDE: str = "VESDE"

    # Model hyperparameters
    time_feature: int = 128
    scale: float = 30.0
    sigma_min: float = 0.3
    sigma_max: float = 10.0
    N: int = 1000

    # UNet hyperparameters
    embedding_dim: int = 128
    base_channels: int = 64
    max_channels: int = 128
    up_down_factor: int = 2
    n_resolution: int = 4
    n_resnet_blocks: int = 2
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    group_norm_size: int = 32
    dropout: float = 0.1
    padding: int = 1
    activation: "str" = "swish"
    skip_rescale: bool = False
    sampling_method: str = "naive"
    fir: bool = False
    fir_kernel_size: int = 3

    # Predictor hyperparameters


class BigParser(SDEDiffusionExperimentParser, SDEDiffusionModelParser):
    pass


class SDEDiffusionTrainer:
    def __init__(
        self, dataset: DiffusionDataset, config: BigParser, logging: bool = False
    ):
        self.config = config
        self.logging = logging
        if logging and (jax.process_index() == 0):
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.as_dict(),
            )

        n_processes = jax.process_count()
        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ("b"))
        self.sharding = jax.sharding.NamedSharding(
            self.global_mesh,
            jax.sharding.PartitionSpec(
                ("b"),
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

        self.data_shape = train_set.dataset.get_shape()

        self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))

        if config.activation == "swish":
            activation = jax.nn.swish
        elif config.activation == "relu":
            activation = jax.nn.relu
        else:
            raise NotImplementedError

        unet_config = UnetConfig(
            num_dim=len(self.data_shape[1:]),
            input_channels=self.data_shape[0],
            output_channels=self.data_shape[0],
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

        unet = Unet(subkey, unet_config, activation=activation)
        self.key, subkey = jax.random.split(self.key)
        subkeys = jax.random.split(subkey, 2)
        time_embed = eqx.nn.Sequential(
            [
                eqx.nn.Linear(config.time_feature, config.time_feature, key=subkeys[0]),
                eqx.nn.Lambda(activation),
                eqx.nn.Linear(
                    config.time_feature, config.embedding_dim, key=subkeys[1]
                ),
            ]
        )

        self.key, subkey = jax.random.split(self.key)
        gaussian_feature = GaussianFourierFeatures(
            config.time_feature, subkey, scale=config.scale
        )
        sde_func = VESDE(
            sigma_min=config.sigma_min, sigma_max=config.sigma_max, N=config.N
        )  # Choosing the sigma drastically affects the training speed

        self.model = ScoreBasedSDE(
            unet,
            gaussian_feature,
            time_embed,
            lambda t: 1.0,#sde_func.marginal_prob(None, t)[1],
            sde_func,
            # corrector=LangevinCorrector(sde_func, lambda x: x, 0.017, 1),
        )

        self.log_model = copy.deepcopy(self.model)

        self.optimizer = optax.chain(optax.adam(config.learning_rate), optax.ema(0.999))
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        if jax.process_index() == 0:
            print("Start training")
        max_loss = 1e10
        self.best_model = self.model
        logging_key = jax.random.PRNGKey(self.config.seed + 1203472)

        if self.logging:
            logging_key, subkey = jax.random.split(logging_key)
            prior = self.model.sde.sample_prior(subkey, self.data_shape)
            wandb.log(
                {
                    "prior_min": np.min(prior),
                    "prior_max": np.max(prior),
                    "prior_mean": np.mean(prior),
                }
            )
            logging_time = jnp.linspace(0, 1, self.config.log_t_step)
            test_example = jnp.array(next(iter(self.test_loader)))[0]
            self.log_norm_check(subkey, self.log_model, test_example)
            logging_key, subkey = jax.random.split(logging_key)
            key, x, x_mean = self.log_model.sample(
                subkey, self.data_shape, self.log_model.sde.N
            )
            fig = plt.figure()
            plt.plot(x.flatten(), label="sample")
            wandb.log({"sample": fig})
            plt.close()

        for step in range(1, self.config.n_epochs + 1):
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
                    train=True,
                )
                self.key, subkey = jax.random.split(self.key)
                _, _, test_loss = self.run_epoch(
                    self.model,
                    self.opt_state,
                    self.test_loader,
                    subkey,
                    step,
                    log_loss=True,
                    train=False,
                )

                if max_loss > test_loss:
                    max_loss = test_loss
                    self.best_model = self.model
                    self.best_model.save_model(self.config.output_path + "/best_model")

                if self.logging:
                    logging_key, subkey = jax.random.split(logging_key)
                    wandb.log(
                        {
                            "training_loss": train_loss,
                            "test_loss": test_loss,
                        },
                        step=step,
                    )
                    test_example = jnp.array(next(iter(self.test_loader)))[0]
                    self.model.save_model(self.config.output_path + "/latest_model")
                    log_model = self.best_model
                    # log_model = self.log_model.load_model(
                    #     self.config.output_path + "/best_model"
                    # )
                    self.log_norm_check(subkey, log_model, test_example)
                    logging_key, subkey = jax.random.split(logging_key)
                    key, x, x_mean = log_model.sample(
                        subkey, self.data_shape, log_model.sde.N
                    )
                    fig = plt.figure()
                    plt.plot(x.flatten(), label="sample")
                    wandb.log({"sample": fig})
                    plt.close()

            else:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.run_epoch(
                    self.model,
                    self.opt_state,
                    self.train_loader,
                    subkey,
                    step,
                    log_loss=False,
                    train=True,
                )

    @staticmethod
    @eqx.filter_jit
    def run_step(
        model: ScoreBasedSDE,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
        opt_update,
        train: bool = True,
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(
            jax.vmap(model.loss)(batch, key)
        )
        if train:
            loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(
                model, batch, keys
            )
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_values
        else:
            loss_values = single_device_loss(model, batch, keys)
            return model, opt_state, loss_values

    def run_epoch(
        self,
        model: ScoreBasedSDE,
        opt_state: PyTree,
        data_loader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
        train: bool = True,
    ) -> tuple[ScoreBasedSDE, PyTree, Array | float]:
        data_loader.sampler.set_epoch(epoch)
        loss = 0
        if self.config.distributed:
            for batch in data_loader:
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
                model, opt_state, loss_values = self.run_step(
                    model,
                    opt_state,
                    global_batch,
                    subkey,
                    self.optimizer.update,
                    train=train,
                )
                if log_loss:
                    loss += jnp.sum(process_allgather(loss_values))
            loss = loss / jax.process_count() / len(data_loader) / np.sum(self.data_shape)
        else:
            for batch in data_loader:
                key, subkey = jax.random.split(key)
                model, opt_state, loss_values = self.run_step(
                    model,
                    opt_state,
                    jnp.array(batch),
                    subkey,
                    self.optimizer.update,
                    train=train,
                )
                if log_loss:
                    loss += jnp.sum(loss_values)
            loss = loss / len(data_loader) / np.sum(self.data_shape)
        return model, opt_state, loss

    def log_norm_check(
        self,
        key: PRNGKeyArray,
        model: ScoreBasedSDE,
        data: Float[Array, " 1 datashape"],
    ):
        key, subkey = jax.random.split(key)

        score_ratio = []

        t_set = jnp.linspace(1, 1e-5, model.sde.N)
        for t in t_set:
            _, sigma_t = model.sde.marginal_prob(data, t)

            x_t = data + sigma_t * jax.random.normal(subkey, shape=(data.shape))

            key, subkey = jax.random.split(key)
            score_slic = eqx.filter_jit(model.score)(x_t, jnp.array([t]), subkey)

            score_gaussian_mag = jnp.sqrt(
                jnp.sum((x_t / (0.2**2 + sigma_t**2)) ** 2)
            )  # jnp.sqrt(x_t.flatten().shape[0])
            score_slic_mag = jnp.sqrt(jnp.sum((score_slic) ** 2))

            score_ratio.append(score_slic_mag / score_gaussian_mag)

        t_set = np.array(t_set)
        score_ratio = np.array(score_ratio)

        fig = plt.figure()
        plt.plot(t_set, score_ratio)
        plt.ylim(0.5, 1.5)
        wandb.log({"score_ratio": fig})
