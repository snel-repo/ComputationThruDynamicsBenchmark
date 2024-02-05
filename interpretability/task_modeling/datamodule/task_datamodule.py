import logging
import os
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from gymnasium import Env
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from interpretability.task_modeling.datamodule.samplers import (
    RandomSampler,
    SequentialSampler,
)


def save_dict_to_pickle(dic, filename):
    """
    Save a dictionary to a pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)


def load_dict_from_pickle(filename):
    """
    Load a dictionary from a pickle file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


logger = logging.getLogger(__name__)


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class TaskDataModule(pl.LightningDataModule):
    """Data module for task training"""

    def __init__(
        self,
        data_env: Env = None,
        n_samples: int = 2000,
        seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """Initialize the data module
        Args:
            data_env (Env): The environment to simulate
            n_samples (int): The number of samples (trials) to simulate
            seed (int): The random seed to use
            batch_size (int): The batch size
            num_workers (int): The number of workers to use for data loading

        """
        super().__init__()
        self.save_hyperparameters()
        # Generate the dataset tag
        self.data_env = data_env
        self.name = None
        self.input_labels = None
        self.output_labels = None

    def set_environment(self, data_env, data_path):
        """Set the environment for the data module"""
        self.data_path = data_path
        self.data_env = data_env

        self.name = (
            f"{data_env.dataset_name}_{self.hparams.n_samples}S_{data_env.n_timesteps}T"
            f"_{self.hparams.seed}seed"
        )
        # if data_env has a noise parameter, add it to the name
        if hasattr(data_env, "noise"):
            self.name += f"_{data_env.noise}"

        # Set input/output labels according to the data environment
        self.input_labels = self.data_env.input_labels
        self.output_labels = self.data_env.output_labels

        # Set extra data if it exists
        if hasattr(self.data_env, "extra"):
            self.extra = self.data_env.extra

        # If the environment has a specific sampling function, use it
        if hasattr(self.data_env, "sampler"):
            self.sampler_func = data_env.sampler
            self.val_sampler_func = SequentialSampler
        else:  # otherwise use the default samplers
            self.sampler_func = RandomSampler
            self.val_sampler_func = SequentialSampler

    def prepare_data(self):
        """Prepare the data for task-training

        All task-environments must have a generate_dataset method,
        called by prepare_data. This method returns a dictionary
        with the following mandatory keys:
            inputs: The inputs to the model
            targets: The targets for the model
                (e.g., the outputs of the environment or MotorNet goal)
            ics: The initial conditions for the environment
                (e.g., the start state for MotorNet)
            conds: The conditions for the task
                (e.g., which task in MultiTask)
            extra: Any extra data that is needed for training
                (e.g., trial length for MultiTask)
        """
        hps = self.hparams

        filename_h5 = f"{self.name}.h5"
        filename_pkl = f"{self.name}.pkl"
        fpath = os.path.join(self.data_path, filename_h5)
        fpath_pkl = os.path.join(self.data_path, filename_pkl)

        # Check if the dataset already exists, and if so, load it
        if os.path.isfile(fpath) and os.path.isfile(fpath_pkl):
            logger.info(f"Loading dataset {self.name}")
            return
        logger.info(f"Generating dataset {self.name}")

        # Simulate the task
        dataset_dict = self.data_env.generate_dataset(self.hparams.n_samples)

        # Extract the inputs, outputs, and initial conditions
        inputs_ds = dataset_dict["inputs"]
        targets_ds = dataset_dict["targets"]
        ics_ds = dataset_dict["ics"]
        conds_ds = dataset_dict["conds"]
        extra_ds = dataset_dict["extra"]

        keys = list(dataset_dict.keys())
        keys.remove("inputs")
        keys.remove("targets")
        keys.remove("ics")
        keys.remove("conds")

        # Perform data splits
        num_trials = ics_ds.shape[0]
        inds = np.arange(num_trials)
        train_inds, valid_inds = train_test_split(
            inds, test_size=0.2, random_state=hps.seed
        )

        # Save the trajectories
        with h5py.File(fpath, "w") as h5file:
            h5file.create_dataset("train_ics", data=ics_ds[train_inds])
            h5file.create_dataset("valid_ics", data=ics_ds[valid_inds])

            h5file.create_dataset("train_inputs", data=inputs_ds[train_inds])
            h5file.create_dataset("valid_inputs", data=inputs_ds[valid_inds])

            h5file.create_dataset("train_targets", data=targets_ds[train_inds])
            h5file.create_dataset("valid_targets", data=targets_ds[valid_inds])

            h5file.create_dataset("train_conds", data=conds_ds[train_inds])
            h5file.create_dataset("valid_conds", data=conds_ds[valid_inds])

            h5file.create_dataset("train_inds", data=train_inds)
            h5file.create_dataset("valid_inds", data=valid_inds)

            h5file.create_dataset("train_extra", data=extra_ds[train_inds])
            h5file.create_dataset("valid_extra", data=extra_ds[valid_inds])

        # Save the dataset dictionary as a pickle file for debugging, etc.
        save_dict_to_pickle(dataset_dict, fpath_pkl)

    def setup(self, stage=None):
        # Load data arrays from file
        data_path = os.path.join(self.data_path, f"{self.name}.h5")
        data_path_pkl = os.path.join(self.data_path, f"{self.name}.pkl")
        with h5py.File(data_path, "r") as h5file:
            train_ics = to_tensor(h5file["train_ics"][()])
            valid_ics = to_tensor(h5file["valid_ics"][()])

            train_inputs = to_tensor(h5file["train_inputs"][()])
            valid_inputs = to_tensor(h5file["valid_inputs"][()])

            train_targets = to_tensor(h5file["train_targets"][()])
            valid_targets = to_tensor(h5file["valid_targets"][()])

            train_conds = to_tensor(h5file["train_conds"][()])
            valid_conds = to_tensor(h5file["valid_conds"][()])

            # Load the indices
            train_inds = to_tensor(h5file["train_inds"][()])
            valid_inds = to_tensor(h5file["valid_inds"][()])
            # test_inds = to_tensor(h5file["test_inds"][()])

            train_extra = to_tensor(h5file["train_extra"][()])
            valid_extra = to_tensor(h5file["valid_extra"][()])

        self.all_data = load_dict_from_pickle(data_path_pkl)

        # Store datasets
        self.train_ds = TensorDataset(
            train_ics,
            train_inputs,
            train_targets,
            train_inds,
            train_conds,
            train_extra,
        )

        self.valid_ds = TensorDataset(
            valid_ics, valid_inputs, valid_targets, valid_inds, valid_conds, valid_extra
        )

        self.train_sampler = self.sampler_func(
            data_source=self.train_ds, num_samples=self.hparams.batch_size
        )
        self.valid_sampler = self.val_sampler_func(
            data_source=self.valid_ds, num_samples=self.hparams.batch_size
        )

    def train_dataloader(self, shuffle=True):
        train_dl = DataLoader(
            self.train_ds,
            batch_sampler=self.train_sampler,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_sampler=self.valid_sampler,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
