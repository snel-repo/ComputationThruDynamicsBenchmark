import pickle
import types

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.analysis import Analysis
from interpretability.comparison.fixedpoints import find_fixed_points


def get_model_inputs_SAE(self):
    dt_train_ds = self.datamodule.train_ds
    dt_val_ds = self.datamodule.valid_ds
    dt_spiking = torch.cat((dt_train_ds.tensors[0], dt_val_ds.tensors[0]), dim=0)
    dt_inputs = torch.cat((dt_train_ds.tensors[2], dt_val_ds.tensors[2]), dim=0)

    return dt_spiking, dt_inputs


def get_model_inputs_LFADS(self):
    dt_train_ds = self.datamodule.train_data
    dt_val_ds = self.datamodule.valid_data

    spiking_train = dt_train_ds[0][0]
    inputs_train = dt_train_ds[0][2]
    spiking_val = dt_val_ds[0][0]
    inputs_val = dt_val_ds[0][2]

    dt_spiking = torch.cat((spiking_train, spiking_val), dim=0)
    dt_inputs = torch.cat((inputs_train, inputs_val), dim=0)
    return dt_spiking, dt_inputs


def get_model_outputs_SAE(self):
    dt_spiking, dt_inputs = self.get_model_inputs()
    rates, latents = self.model(dt_spiking, dt_inputs)
    return rates, latents


def get_model_outputs_LFADS(self):
    t_data = self.datamodule.train_data
    v_data = self.datamodule.valid_data
    output_dict = self.model(t_data[0])
    output_dict_val = self.model(v_data[0])
    rates_train = output_dict[0]
    latents_train = output_dict[6]
    rates_val = output_dict_val[0]
    latents_val = output_dict_val[6]
    rates = torch.cat((rates_train, rates_val), dim=0)
    latents = torch.cat((latents_train, latents_val), dim=0)
    return rates, latents


def get_latents_SAE(self):
    _, latents = self.get_model_outputs()
    return latents


def get_latents_LFADS(self):
    rates, latents = self.get_model_outputs()
    return latents


def get_dynamics_model_SAE(self):
    return self.model.decoder.cell


def get_dynamics_model_LFADS(self):
    return self.model.decoder.rnn.cell.gen_cell


class Analysis_DT(Analysis):
    def __init__(self, run_name, filepath, model_type="SAE"):
        self.tt_or_dt = "dt"
        self.run_name = run_name
        self.model_type = model_type
        self.load_wrapper(filepath)
        if self.model_type == "SAE":
            self.get_model_inputs = types.MethodType(get_model_inputs_SAE, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_SAE, self)
            self.get_latents = types.MethodType(get_latents_SAE, self)
            self.get_dynamics_model = types.MethodType(get_dynamics_model_SAE, self)
        elif self.model_type == "LFADS":
            self.get_model_inputs = types.MethodType(get_model_inputs_LFADS, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_LFADS, self)
            self.get_latents = types.MethodType(get_latents_LFADS, self)
            self.get_dynamics_model = types.MethodType(get_dynamics_model_LFADS, self)

    def load_wrapper(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.datamodule = pickle.load(f)

    def compute_FPs(
        self,
        noiseless=True,
        inputs=None,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
    ):
        # Compute latent activity from task trained model
        if inputs is None and noiseless:
            _, inputs = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()

        fps = find_fixed_points(
            model=self.get_dynamics_model(),
            state_trajs=latents,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        return fps

    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cuda",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
    ):

        latents = self.get_latents().detach().numpy()
        fps = self.compute_FPs(
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        xstar = fps.xstar
        q_vals = fps.qstar
        is_stable = fps.is_stable
        q_flag = q_vals < q_thresh
        pca = PCA(n_components=3)
        xstar_pca = pca.fit_transform(xstar)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.transform(lats_flat)
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        # Make a color vector based on stability
        colors = np.zeros((xstar.shape[0], 3))
        colors[is_stable, 0] = 1
        colors[~is_stable, 2] = 1
        ax.scatter(xstar_pca[q_flag, 0], xstar_pca[q_flag, 1], xstar_pca[q_flag, 2])
        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title(f"{self.model_type}_Fixed Points")
        plt.show()
        plt.savefig(self.run_name + f"_{self.model_type}_fps.png")
