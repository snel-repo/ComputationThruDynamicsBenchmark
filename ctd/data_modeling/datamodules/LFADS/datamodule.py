import os

import dotenv
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .tuples import SessionBatch

dotenv.load_dotenv(override=True)

MANDATORY_KEYS = {
    "train": ["encod_data", "recon_data"],
    "valid": ["encod_data", "recon_data"],
    "test": ["encod_data", "recon_data"],
}
HOME_DIR = os.environ.get("HOME_DIR")


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


def attach_tensors(datamodule, data_dict: dict, extra_keys: list[str] = []):
    hps = datamodule.hparams
    sv_gen = torch.Generator().manual_seed(hps.sv_seed)

    def create_session_batch(prefix):
        assert all(f"{prefix}_{key}" in data_dict for key in MANDATORY_KEYS[prefix])
        encod_data = to_tensor(data_dict[f"{prefix}_encod_data"])
        recon_data = to_tensor(data_dict[f"{prefix}_recon_data"])
        n_samps, n_steps, _ = encod_data.shape
        if hps.sv_rate > 0:
            # Create sample validation mask # TODO: Sparse and use complement?
            bern_p = 1 - hps.sv_rate if prefix != "test" else 1.0
            sv_mask = (torch.rand(encod_data.shape, generator=sv_gen) < bern_p).float()
        else:
            # Create a placeholder tensor
            sv_mask = torch.ones(n_samps, 0, 0)
        # Load or simulate external inputs
        if f"{prefix}_inputs" in data_dict and datamodule.hparams.provide_inputs:
            ext_input = to_tensor(data_dict[f"{prefix}_inputs"])
        else:
            ext_input = torch.zeros(n_samps, n_steps, 0)
        if f"{prefix}_truth" in data_dict:
            # Load or simulate ground truth TODO: use None instead of NaN?
            cf = data_dict["conversion_factor"]
            truth = to_tensor(data_dict[f"{prefix}_truth"]) / cf
        else:
            truth = torch.full((n_samps, 0, 0), float("nan"))
        # Remove unnecessary data during IC encoder segment
        sv_mask = sv_mask[:, hps.dm_ic_enc_seq_len :]
        ext_input = ext_input[:, hps.dm_ic_enc_seq_len :]
        truth = truth[:, hps.dm_ic_enc_seq_len :, :]
        # Extract data for any extra keys
        other = [to_tensor(data_dict[f"{prefix}_{k}"]) for k in extra_keys]
        return (
            SessionBatch(
                encod_data=encod_data,
                recon_data=recon_data,
                ext_input=ext_input,
                truth=truth,
                sv_mask=sv_mask,
            ),
            tuple(other),
        )

    # import pdb; pdb.set_trace()
    # Store the datasets on the datamodule
    datamodule.train_data = create_session_batch("train")
    datamodule.train_ds = SessionDataset(*datamodule.train_data)
    datamodule.valid_data = create_session_batch("valid")
    datamodule.valid_ds = SessionDataset(*datamodule.valid_data)
    if "test_encod_data" in data_dict:
        datamodule.test_data = create_session_batch("test")
        datamodule.test_ds = SessionDataset(*datamodule.test_data)


def reshuffle_train_valid(data_dict, seed, ratio=None):
    # Identify the data to be reshuffled
    data_keys = [k.replace("train_", "") for k in data_dict.keys() if "train_" in k]
    # Combine all training and validation data arrays
    arrays = [
        np.concatenate([data_dict["train_" + k], data_dict["valid_" + k]])
        for k in data_keys
    ]
    # Reshuffle and split training and validation data
    valid_size = ratio if ratio is not None else len(data_dict["valid_" + data_keys[0]])
    arrays = train_test_split(*arrays, test_size=valid_size, random_state=seed)
    train_arrays = [a for i, a in enumerate(arrays) if (i - 1) % 2]
    valid_arrays = [a for i, a in enumerate(arrays) if i % 2]
    # Replace the previous data with the newly split data
    for k, ta, va in zip(data_keys, train_arrays, valid_arrays):
        data_dict.update({"train_" + k: ta, "valid_" + k: va})
    return data_dict


class SessionDataset(Dataset):
    def __init__(
        self, model_tensors: SessionBatch[Tensor], extra_tensors: tuple[Tensor]
    ):
        all_tensors = [*model_tensors, *extra_tensors]
        assert all(
            all_tensors[0].size(0) == tensor.size(0) for tensor in all_tensors
        ), "Size mismatch between tensors"
        self.model_tensors = model_tensors
        self.extra_tensors = extra_tensors

    def __getitem__(self, index):
        model_tensors = SessionBatch(*[t[index] for t in self.model_tensors])
        extra_tensors = tuple(t[index] for t in self.extra_tensors)
        return model_tensors, extra_tensors

    def __len__(self):
        return len(self.model_tensors[0])


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prefix: str,
        system: str,
        gen_model: str,
        n_neurons: int,
        nonlin_embed: bool,
        obs_noise: str,
        seed: int,
        batch_keys: list[str] = [],
        attr_keys: list[str] = [],
        batch_size: int = 64,
        reshuffle_tv_seed: int = None,
        reshuffle_tv_ratio: float = None,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
        provide_inputs: bool = True,
        file_index: int = 0,
    ):
        assert (
            reshuffle_tv_seed is None or len(attr_keys) == 0
        ), "Dataset reshuffling is incompatible with the `attr_keys` argument."
        super().__init__()

        filedir = (
            f"{prefix}_"
            f"{system}_"
            f"model_{gen_model}_"
            f"n_neurons_{n_neurons}_"
            f"seed_{seed}"
        )
        data_dir = os.path.join(HOME_DIR, "content", "datasets", "dt")
        fpath = os.path.join(data_dir, filedir)
        dirs = os.listdir(fpath)
        if file_index >= len(dirs):
            raise ValueError(
                f"File index {file_index} is out of range for directory {fpath}"
            )
        else:
            filename = dirs[file_index]
            self.name = filename

        self.fpath = os.path.join(fpath, filename)
        self.save_hyperparameters()

    def setup(self, stage=None):
        hps = self.hparams

        # Load data arrays from the file
        with h5py.File(self.fpath, "r") as h5file:
            data_dict = {k: v[()] for k, v in h5file.items()}

        if hps.reshuffle_tv_seed is not None:
            data_dict = reshuffle_train_valid(
                data_dict, hps.reshuffle_tv_seed, hps.reshuffle_tv_ratio
            )

        attach_tensors(self, data_dict, extra_keys=hps.batch_keys)
        for attr_key in hps.attr_keys:
            setattr(self, attr_key, data_dict[attr_key])

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        # NOTE: Returning dicts of DataLoaders is incompatible with trainer.predict,
        # but convenient for posterior sampling. Can't use CombinedLoader here because
        # we only want to see each sample once.
        dataloader = {
            "train": DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
            ),
            "valid": DataLoader(
                self.valid_ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
            ),
        }

        # Add the test dataset if it is available
        if hasattr(self, "test_ds"):
            dataloader["test"] = DataLoader(
                self.test_ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
            )
        return dataloader
