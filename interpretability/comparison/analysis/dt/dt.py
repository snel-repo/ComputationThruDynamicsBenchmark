import pickle
import types

import torch
from jax import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.analysis import Analysis
from interpretability.comparison.fixedpoints import find_fixed_points


def get_model_inputs_LDS(self):
    train_ds = torch.tensor(self.datamodule.train_data)
    val_ds = torch.tensor(self.datamodule.eval_data)
    train_inputs = torch.tensor(self.datamodule.train_inputs)
    val_inputs = torch.tensor(self.datamodule.eval_inputs)
    dt_spiking = torch.cat((train_ds, val_ds), dim=0)
    dt_inputs = torch.cat((train_inputs, val_inputs), dim=0)
    return dt_spiking, dt_inputs


def get_model_inputs_SAE(self):
    dt_train_ds = self.datamodule.train_ds
    dt_val_ds = self.datamodule.valid_ds
    dt_spiking = torch.cat((dt_train_ds.tensors[0], dt_val_ds.tensors[0]), dim=0)
    dt_inputs = torch.cat((dt_train_ds.tensors[2], dt_val_ds.tensors[2]), dim=0)

    return dt_spiking, dt_inputs


def get_true_rates_SAE(self):
    dt_train_ds = self.datamodule.train_ds
    dt_val_ds = self.datamodule.valid_ds
    rates_train = dt_train_ds.tensors[6]
    rates_val = dt_val_ds.tensors[6]
    true_rates = torch.cat((rates_train, rates_val), dim=0)
    return true_rates


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


def get_model_outputs_LDS(self):
    spiking, inputs = self.get_model_inputs()
    key = random.PRNGKey(0)

    out_dict = self.model.forward(spiking, inputs, key)
    return out_dict["lograte_t"], out_dict["gen_t"]


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


def get_latents_LDS(self):
    _, latents = self.get_model_outputs()
    return latents


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
            self.get_true_rates = types.MethodType(get_true_rates_SAE, self)
        elif self.model_type == "LFADS":
            self.get_model_inputs = types.MethodType(get_model_inputs_LFADS, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_LFADS, self)
            self.get_latents = types.MethodType(get_latents_LFADS, self)
            self.get_dynamics_model = types.MethodType(get_dynamics_model_LFADS, self)
            self.get_true_rates = None
        elif self.model_type == "LDS":
            self.get_model_inputs = types.MethodType(get_model_inputs_LDS, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_LDS, self)
            self.get_latents = types.MethodType(get_latents_LDS, self)
            self.get_tru_rates = None
            self.get_dynamics_model = None

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
        latents = latents.to(device)
        inputs = inputs.to(device)

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

        xstar_pca = xstar_pca[q_flag]
        is_stable = is_stable[q_flag]

        ax.scatter(
            xstar_pca[is_stable, 0],
            xstar_pca[is_stable, 1],
            xstar_pca[is_stable, 2],
            c="g",
        )
        ax.scatter(
            xstar_pca[~is_stable, 0],
            xstar_pca[~is_stable, 1],
            xstar_pca[~is_stable, 2],
            c="r",
        )

        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title(f"{self.model_type}_Fixed Points")
        plt.show()

    def plot_trial(self, num_trials=10, scatterPlot=True):
        latents = self.get_latents().detach().numpy()
        pca = PCA(n_components=3)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.fit_transform(lats_flat)
        lats_pca = lats_pca.reshape(-1, latents.shape[1], 3)
        if scatterPlot:

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for i in range(num_trials):
                ax.plot(
                    lats_pca[i, :, 0],
                    lats_pca[i, :, 1],
                    lats_pca[i, :, 2],
                )
            ax.set_title(f"{self.model_type}_Trial Latent Activity")
        else:
            fig = plt.figure(figsize=(10, 4 * num_trials))
            for i in range(num_trials):
                ax = fig.add_subplot(num_trials, 1, i + 1)
                ax.plot(lats_pca[i, :, 0])
                ax.plot(lats_pca[i, :, 1])
                ax.plot(lats_pca[i, :, 2])
            ax.set_title(f"{self.model_type}_Trial Latent Activity")

        plt.show()

    def get_inputs(self):
        _, inputs = self.get_model_inputs()
        return inputs
