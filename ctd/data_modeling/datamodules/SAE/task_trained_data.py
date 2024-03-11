import logging
import os

import dotenv
import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
dotenv.load_dotenv(override=True)
HOME_DIR = os.environ.get("HOME_DIR")


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class TaskTrainedRNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prefix=None,
        system: str = "3BFF",
        gen_model: str = "GRU_RNN",
        n_neurons: int = 50,
        nonlin_embed: bool = False,
        seed: int = 0,
        obs_noise: str = "poisson",
        batch_size: int = 64,
        num_workers: int = 2,
        provide_inputs: bool = True,
        file_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.data_dir = os.path.join(HOME_DIR, "content", "datasets", "dt")

        filedir = (
            f"{prefix}_"
            f"{system}_"
            f"model_{gen_model}_"
            f"n_neurons_{n_neurons}_"
            f"seed_{seed}"
        )
        fpath = os.path.join(self.data_dir, filedir)
        dirs = os.listdir(fpath)
        if file_index >= len(dirs):
            raise ValueError(
                f"File index {file_index} is out of range for directory {fpath}"
            )
        else:
            filename = dirs[0]

        self.name = os.path.join(filename)
        self.fpath = filedir

    def prepare_data(self):
        filename = self.name
        fpath = os.path.join(self.data_dir, self.fpath, filename)
        if os.path.isfile(fpath):
            logger.info(f"Loading dataset {self.name}")
            return
        else:
            # throw an error here
            raise FileNotFoundError(f"Dataset {self.name} not found at {self.fpath}")

    def setup(self, stage=None):
        # Load data arrays from file
        data_path = os.path.join(self.data_dir, self.fpath, self.name)
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

            train_inputs = h5file["train_inputs"][()]
            valid_inputs = h5file["valid_inputs"][()]

            train_extra = h5file["train_extra"][()]
            valid_extra = h5file["valid_extra"][()]

            # self.test_inputs = h5file["test_inputs"][()]

        train_inputs = to_tensor(train_inputs)
        valid_inputs = to_tensor(valid_inputs)

        train_extra = to_tensor(train_extra)
        valid_extra = to_tensor(valid_extra)

        if self.hparams.provide_inputs:
            # Store datasets
            self.train_ds = TensorDataset(
                train_data,
                train_data,
                train_inputs,
                train_extra,
                train_latents,
                train_inds,
                train_activity,
            )

            self.valid_ds = TensorDataset(
                valid_data,
                valid_data,
                valid_inputs,
                valid_extra,
                valid_latents,
                valid_inds,
                valid_activity,
            )
        else:
            self.train_ds = TensorDataset(
                train_data,
                train_data,
                None,
                train_extra,
                train_latents,
                train_inds,
                train_activity,
            )

            self.valid_ds = TensorDataset(
                valid_data,
                valid_data,
                None,
                valid_extra,
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
