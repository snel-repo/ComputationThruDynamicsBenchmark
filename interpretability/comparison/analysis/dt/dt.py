import pickle

import numpy as np
import torch

from interpretability.comparison.analysis.analysis import Analysis


class Analysis_DT(Analysis):
    def __init__(self, run_name, filepath):
        self.tt_or_dt = "dt"
        self.run_name = run_name
        self.load_wrapper(filepath)

    def load_wrapper(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.datamodule = pickle.load(f)

    def get_model_input(self):
        dt_train_ds = self.datamodule.train_ds
        dt_val_ds = self.datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4].int()
        dt_val_inds = dt_val_ds.tensors[4].int()

        n_t, n_neurons = dt_train_ds.tensors[0][0].shape
        _, n_inputs = dt_train_ds.tensors[2][0].shape

        trial_count = dt_train_inds.shape[0] + dt_val_inds.shape[0]

        dt_val_ds = self.datamodule.valid_ds
        dt_spiking = np.zeros((trial_count, n_t, n_neurons))
        dt_inputs = np.zeros((trial_count, n_t, n_inputs))
        for i in range(len(dt_train_inds)):
            dt_spiking[dt_train_inds[i], :, :] = dt_train_ds.tensors[0][i]
            dt_inputs[dt_train_inds[i], :, :] = dt_train_ds.tensors[2][i]
        for i in range(len(dt_val_inds)):
            dt_spiking[dt_val_inds[i], :, :] = dt_val_ds.tensors[0][i]
            dt_inputs[dt_val_inds[i], :, :] = dt_val_ds.tensors[2][i]

        dt_spiking = torch.tensor(dt_spiking).float()
        dt_inputs = torch.tensor(dt_inputs).float()
        return dt_spiking, dt_inputs

    def get_model_output(self):
        dt_spiking, dt_inputs = self.get_model_input()
        rates, latents = self.model(dt_spiking, dt_inputs)
        return rates, latents

    def get_latents(self):
        _, latents = self.get_model_output()
        return latents
