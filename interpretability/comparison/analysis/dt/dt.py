import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.analysis import Analysis
from interpretability.comparison.fixedpoints import find_fixed_points


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
            _, inputs = self.get_model_input()
            latents = self.get_latents()
        elif inputs is None and not noiseless:
            _, inputs, _ = self.get_model_input()
            latents = self.get_latents()
        else:
            latents = self.get_latents()

        fps = find_fixed_points(
            model=self.model,
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
        device="cpu",
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
        plt.show()
