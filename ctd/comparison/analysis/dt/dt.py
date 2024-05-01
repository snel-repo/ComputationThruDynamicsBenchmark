import io
import pickle
import types

import numpy as np
import torch
from jax import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ctd.comparison.analysis.analysis import Analysis
from ctd.comparison.fixedpoints import find_fixed_points
from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def get_trial_lens_SAE(self, phase="all"):
    if phase == "all":
        dt_train_ds = self.datamodule.train_ds
        dt_val_ds = self.datamodule.valid_ds
        trial_lens = torch.cat(
            (dt_train_ds.tensors[3][:, -1], dt_val_ds.tensors[3][:, -1]), dim=0
        )
    elif phase == "train":
        trial_lens = self.datamodule.train_ds.tensors[3][:, -1]
    elif phase == "val":
        trial_lens = self.datamodule.valid_ds.tensors[3][:, -1]
    return trial_lens


def get_trial_lens_LFADS(self, phase="all"):
    dt_extra = []
    if phase == "all":
        train_dl = self.datamodule.train_dataloader(shuffle=False)
        val_dl = self.datamodule.val_dataloader()
        for batch in train_dl:
            # Move data to the right device
            train_extra = batch[1][3][:, -1]
            dt_extra.append(train_extra)
        for batch in val_dl:
            # Move data to the right device
            val_extra = batch[1][3][:, -1]
            dt_extra.append(val_extra)
    elif phase == "train":
        train_dl = self.datamodule.train_dataloader(shuffle=False)
        for batch in train_dl:
            # Move data to the right device
            train_extra = batch[1][3][:, -1]
            dt_extra.append(train_extra)
    elif phase == "val":
        val_dl = self.datamodule.val_dataloader()
        for batch in val_dl:
            # Move data to the right device
            val_extra = batch[1][3][:, -1]
            dt_extra.append(val_extra)
    dt_extra = torch.cat(dt_extra, dim=0)
    return dt_extra


def get_model_inputs_LDS(self):
    train_ds = torch.tensor(self.datamodule.train_data)
    val_ds = torch.tensor(self.datamodule.eval_data)
    train_inputs = torch.tensor(self.datamodule.train_inputs)
    val_inputs = torch.tensor(self.datamodule.eval_inputs)
    dt_spiking = torch.cat((train_ds, val_ds), dim=0)
    dt_inputs = torch.cat((train_inputs, val_inputs), dim=0)
    return dt_spiking, dt_inputs


def get_model_inputs_SAE(self, phase="all"):
    if phase == "all":
        dt_train_ds = self.datamodule.train_ds
        dt_val_ds = self.datamodule.valid_ds
        dt_spiking = torch.cat((dt_train_ds.tensors[0], dt_val_ds.tensors[0]), dim=0)
        dt_inputs = torch.cat((dt_train_ds.tensors[2], dt_val_ds.tensors[2]), dim=0)
    elif phase == "train":
        dt_spiking = self.datamodule.train_ds.tensors[0]
        dt_inputs = self.datamodule.train_ds.tensors[2]
    elif phase == "val":
        dt_spiking = self.datamodule.valid_ds.tensors[0]
        dt_inputs = self.datamodule.valid_ds.tensors[2]

    return dt_spiking, dt_inputs


def get_true_rates_SAE(self, phase="all"):
    if phase == "all":
        dt_train_ds = self.datamodule.train_ds
        dt_val_ds = self.datamodule.valid_ds
        rates_train = dt_train_ds.tensors[6]
        rates_val = dt_val_ds.tensors[6]
        true_rates = torch.cat((rates_train, rates_val), dim=0)
    elif phase == "train":
        true_rates = self.datamodule.train_ds.tensors[6]
    elif phase == "val":
        true_rates = self.datamodule.valid_ds.tensors[6]
    return true_rates


def get_rates_SAE(self, phase="all"):
    rates, _ = self.get_model_outputs(phase=phase)
    return rates


def get_model_inputs_LFADS(self, phase="all"):
    if phase == "all":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        val_dataloader = self.datamodule.val_dataloader()
        dt_spiking = []
        dt_inputs = []
        for batch in train_ds:
            # Move data to the right device
            spiking_train = batch[0][0]
            inputs_train = batch[0][2]
            dt_spiking.append(spiking_train)
            dt_inputs.append(inputs_train)
        for batch in val_dataloader:
            # Move data to the right device
            spiking_val = batch[0][0]
            inputs_val = batch[0][2]
            dt_spiking.append(spiking_val)
            dt_inputs.append(inputs_val)
    elif phase == "train":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        dt_spiking = []
        dt_inputs = []
        for batch in train_ds:
            # Move data to the right device
            spiking_train = batch[0][0]
            inputs_train = batch[0][2]
            dt_spiking.append(spiking_train)
            dt_inputs.append(inputs_train)
    elif phase == "val":
        val_dataloader = self.datamodule.val_dataloader()
        dt_spiking = []
        dt_inputs = []
        for batch in val_dataloader:
            # Move data to the right device
            spiking_val = batch[0][0]
            inputs_val = batch[0][2]
            dt_spiking.append(spiking_val)
            dt_inputs.append(inputs_val)
    dt_spiking = torch.cat(dt_spiking, dim=0)
    dt_inputs = torch.cat(dt_inputs, dim=0)
    return dt_spiking, dt_inputs


def get_true_rates_LFADS(self, phase="all"):
    if phase == "all":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        val_dataloader = self.datamodule.val_dataloader()
        dt_rates = []
        for batch in train_ds:
            # Move data to the right device
            rates_train = batch[1][2]
            dt_rates.append(rates_train)
        for batch in val_dataloader:
            # Move data to the right device
            rates_val = batch[1][2]
            dt_rates.append(rates_val)
    elif phase == "train":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        dt_rates = []
        for batch in train_ds:
            # Move data to the right device
            rates_train = batch[1][2]
            dt_rates.append(rates_train)
    elif phase == "val":
        val_dataloader = self.datamodule.val_dataloader()
        dt_rates = []
        for batch in val_dataloader:
            # Move data to the right device
            rates_val = batch[1][2]
            dt_rates.append(rates_val)
    dt_rates = torch.cat(dt_rates, dim=0)
    return dt_rates


def get_model_outputs_LDS(self):
    spiking, inputs = self.get_model_inputs()
    key = random.PRNGKey(0)

    out_dict = self.model.forward(spiking, inputs, key)
    return out_dict["lograte_t"], out_dict["gen_t"]


def get_model_outputs_SAE(self, phase="all"):
    dt_spiking, dt_inputs = self.get_model_inputs(phase=phase)
    dt_spiking = dt_spiking.to(self.model.device)
    dt_inputs = dt_inputs.to(self.model.device)
    log_rates, latents = self.model(dt_spiking, dt_inputs)
    return torch.exp(log_rates), latents


def get_model_outputs_LFADS(self, phase="all"):
    if phase == "all":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        val_dataloader = self.datamodule.val_dataloader()
        dt_rates = []
        dt_latents = []

        for batch in train_ds:
            # Move data to the right device
            batch = send_batch_to_device(batch, self.model.device)
            # Compute model output
            output = self.model.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            dt_rates.append(output[0])
            dt_latents.append(output[6])

        for batch in val_dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, self.model.device)
            # Compute model output
            output = self.model.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            dt_rates.append(output[0])
            dt_latents.append(output[6])
    elif phase == "train":
        train_ds = self.datamodule.train_dataloader(shuffle=False)
        dt_rates = []
        dt_latents = []
        for batch in train_ds:
            # Move data to the right device
            batch = send_batch_to_device(batch, self.model.device)
            # Compute model output
            output = self.model.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            dt_rates.append(output[0])
            dt_latents.append(output[6])
    elif phase == "val":
        val_dataloader = self.datamodule.val_dataloader()
        dt_rates = []
        dt_latents = []
        for batch in val_dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, self.model.device)
            # Compute model output
            output = self.model.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            dt_rates.append(output[0])
            dt_latents.append(output[6])
    dt_rates = torch.cat(dt_rates, dim=0)
    dt_latents = torch.cat(dt_latents, dim=0)

    return dt_rates, dt_latents


def get_rates_LFADS(self, phase="all"):
    rates, _ = self.get_model_outputs(phase=phase)
    return rates


def get_latents_LDS(self):
    _, latents = self.get_model_outputs()
    return latents


def get_latents_SAE(self, phase="all"):
    _, latents = self.get_model_outputs(phase=phase)
    return latents


def get_latents_LFADS(self, phase="all"):
    rates, latents = self.get_model_outputs(phase=phase)
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
            self.get_rates = types.MethodType(get_rates_SAE, self)
            self.get_trial_lens = types.MethodType(get_trial_lens_SAE, self)
        elif self.model_type == "LFADS":
            self.get_model_inputs = types.MethodType(get_model_inputs_LFADS, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_LFADS, self)
            self.get_latents = types.MethodType(get_latents_LFADS, self)
            self.get_dynamics_model = types.MethodType(get_dynamics_model_LFADS, self)
            self.get_true_rates = types.MethodType(get_true_rates_LFADS, self)
            self.get_rates = types.MethodType(get_rates_LFADS, self)
            self.get_trial_lens = types.MethodType(get_trial_lens_LFADS, self)
        elif self.model_type == "LDS":
            self.get_model_inputs = types.MethodType(get_model_inputs_LDS, self)
            self.get_model_outputs = types.MethodType(get_model_outputs_LDS, self)
            self.get_latents = types.MethodType(get_latents_LDS, self)
            self.get_tru_rates = None
            self.get_dynamics_model = None
        else:
            raise ValueError("Invalid model type! Use 'SAE', 'LFADS', or 'LDS'")

    def load_wrapper(self, filepath):
        if torch.cuda.is_available():
            with open(filepath + "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
        else:
            with open(filepath + "model.pkl", "rb") as f:
                self.model = CPU_Unpickler(f).load()
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = CPU_Unpickler(f).load()

    def to_device(self, device):
        self.model.to(device)
        self.datamodule.to(device)

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
        m_device = self.model.device
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
        self.model.to(m_device)
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
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        zero_flag = q_vals == 0
        q_vals[zero_flag] = 1e-15
        ax.hist(np.log10(q_vals), bins=100)
        ax.set_xlabel("log10(q)")
        ax.set_ylabel("Count")
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
        return fps

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

    def plot_rates(self, phase="val", neurons=[0], n_trials=5):
        gru_rates = self.get_rates(phase=phase)
        true_rates = self.get_true_rates(phase=phase)
        trial_lens = self.get_trial_lens(phase=phase)
        rates_stack = []
        true_rates_stack = []
        for i in range(len(trial_lens)):
            rates_stack.append(gru_rates[i][:].detach().cpu().numpy())
            true_rates_stack.append(true_rates[i][:].detach().cpu().numpy())

        fig, ax = plt.subplots(n_trials, len(neurons), figsize=(10, 10))
        for i in range(n_trials):
            for j in range(len(neurons)):
                neuron = neurons[j]
                if i == 0 and j == 0:
                    ax[i, j].plot(
                        rates_stack[i][:, neuron],
                        color="black",
                        label="Estimated Rates",
                    )
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron],
                        color="black",
                        linestyle="--",
                        label="True Rates",
                    )
                else:
                    ax[i, j].plot(rates_stack[i][:, neuron], color="black")
                    # Restart the color order
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron], color="black", linestyle="--"
                    )
                if i == 0:
                    ax[i, j].set_title(f"Neuron {neuron}")
        ax[0, 0].legend()
        plt.show()
