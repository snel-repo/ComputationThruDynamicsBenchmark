import logging
import os

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
from gymnasium import Env
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Load the project data home
dotenv.load_dotenv(override=True)
DATA_HOME = os.environ["TASK_TRAINED_DATA_HOME"]


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class TaskDataModuleCoupled(pl.LightningDataModule):
    def __init__(
        self,
        data_env: Env,
        n_samples: int = 2000,
        seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_env = data_env
        self.save_hyperparameters()
        # Generate the dataset tag

    def set_environment(self, data_env: Env):
        pass

    def prepare_data(self):
        hps = self.hparams

        # Standardize and record original mean and standard deviations
        # Perform data splits
        # Make a tensor of dimension (n_samples) filled with batch_size entries
        inds = np.arange(hps.n_samples)
        # Generate a set of initial conditions from the data env

        joint_list = []
        goal_list = []

        for i in range(hps.n_samples):
            obs, info = self.data_env.reset()
            joint_list.append(torch.squeeze(info["states"]["joint"]))
            goal_list.append(torch.squeeze(info["goal"]))

        joint_list = torch.stack(joint_list, axis=0)
        goal_list = torch.stack(goal_list, axis=0)

        train_inds, valid_inds = train_test_split(
            inds, test_size=0.2, random_state=hps.seed
        )
        self.train_joints = joint_list[train_inds]
        self.train_goals = goal_list[train_inds]
        self.valid_joints = joint_list[valid_inds]
        self.valid_goals = goal_list[valid_inds]

    def setup(self, stage=None):
        # Store datasets
        self.train_ds = TensorDataset(
            self.train_joints,
            self.train_goals,
        )
        self.valid_ds = TensorDataset(
            self.valid_joints,
            self.valid_goals,
        )

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

    # def test_dataloader(self):
    #     test_dl = DataLoader(
    #         self.test_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #     )
    #     return test_dl
