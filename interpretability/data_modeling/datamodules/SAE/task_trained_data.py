import logging
import os

import dotenv
import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Load the project data home
dotenv.load_dotenv(override=True)
DATA_HOME = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/data_modeling/datasets"
)


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


# 3BFF_model_RNN_n_neurons_50_nonlin_embed_False_obs_noise_poisson_seed_0.h5
class TaskTrainedRNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: str = "3BitFlipFlop",
        gen_model: str = "GRU",
        n_neurons: int = 50,
        nonlin_embed: bool = False,
        seed: int = 0,
        obs_noise: str = "poisson",
        batch_size: int = 64,
        num_workers: int = 2,
        provide_inputs: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed

        filename = (
            f"{system}_"
            f"model_{gen_model}_"
            f"n_neurons_{n_neurons}_"
            f"nonlin_embed_{nonlin_embed}_"
            f"obs_noise_{obs_noise}_"
            f"seed_{seed}.h5"
        )
        self.name = filename
        self.fpath = os.path.join(DATA_HOME, filename)

    def prepare_data(self):
        filename = self.name
        fpath = os.path.join(DATA_HOME, filename)
        if os.path.isfile(fpath):
            logger.info(f"Loading dataset {self.name}")
            return
        else:
            # throw an error here
            raise FileNotFoundError(f"Dataset {self.name} not found at {self.fpath}")

    def setup(self, stage=None):
        # Load data arrays from file
        data_path = os.path.join(DATA_HOME, self.name)
        with h5py.File(data_path, "r") as h5file:
            # Load the data
            train_data = to_tensor(h5file["train_encod_data"][()])
            valid_data = to_tensor(h5file["valid_encod_data"][()])
            # test_data = to_tensor(h5file["test_data"][()])
            # Load the activity
            train_activity = to_tensor(h5file["train_activity"][()])
            valid_activity = to_tensor(h5file["valid_activity"][()])
            # test_activity = to_tensor(h5file["test_activity"][()])
            # Load the latents
            train_latents = to_tensor(h5file["train_latents"][()])
            valid_latents = to_tensor(h5file["valid_latents"][()])
            # test_latents = to_tensor(h5file["test_latents"][()])
            # Load the indices
            train_inds = to_tensor(h5file["train_inds"][()])
            valid_inds = to_tensor(h5file["valid_inds"][()])
            # test_inds = to_tensor(h5file["test_inds"][()])
            # Load other parameters
            self.orig_mean = h5file["orig_mean"][()]
            self.orig_std = h5file["orig_std"][()]
            self.readout = h5file["readout"][()]

            self.train_inputs = h5file["train_inputs"][()]
            self.valid_inputs = h5file["valid_inputs"][()]
            # self.test_inputs = h5file["test_inputs"][()]

        train_inputs = to_tensor(self.train_inputs)
        valid_inputs = to_tensor(self.valid_inputs)
        # test_inputs = to_tensor(self.test_inputs)
        if self.hparams.provide_inputs:
            # Store datasets
            self.train_ds = TensorDataset(
                train_data,
                train_data,
                train_inputs,
                train_latents,
                train_inds,
                train_activity,
            )

            self.valid_ds = TensorDataset(
                valid_data,
                valid_data,
                valid_inputs,
                valid_latents,
                valid_inds,
                valid_activity,
            )
        else:
            self.train_ds = TensorDataset(
                train_data,
                train_data,
                None,
                train_latents,
                train_inds,
                train_activity,
            )

            self.valid_ds = TensorDataset(
                valid_data,
                valid_data,
                None,
                valid_latents,
                valid_inds,
                valid_activity,
            )

        # self.test_ds = TensorDataset(
        #     test_data, test_data, test_inputs, test_latents, test_inds, test_activity
        # )

    def train_dataloader(self, shuffle=True):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
