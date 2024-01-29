import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.analysis import Analysis
from interpretability.comparison.fixedpoints import find_fixed_points


class Analysis_TT(Analysis):
    def __init__(self, run_name, filepath):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath)

    def load_wrapper(self, filepath):
        # if self.task_train_wrapper is  empty, load the first one
        with open(filepath + "model.pkl", "rb") as f:
            self.wrapper = pickle.load(f)
        self.env = self.wrapper.task_env
        self.model = self.wrapper.model
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.datamodule = pickle.load(f)
            self.datamodule.prepare_data()
            self.datamodule.setup()
        with open(filepath + "simulator.pkl", "rb") as f:
            self.simulator = pickle.load(f)
        self.task_name = self.datamodule.data_env.dataset_name

    def get_model_input(self):

        all_data = self.datamodule.all_data
        tt_ics = torch.Tensor(all_data["ics"])
        tt_inputs = torch.Tensor(all_data["inputs"])
        tt_targets = torch.Tensor(all_data["targets"])
        return tt_ics, tt_inputs, tt_targets

    def get_model_output(self):
        tt_ics, tt_inputs, tt_targets = self.get_model_input()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_latents(self):
        out_dict = self.get_model_output()
        return out_dict["latents"]

    def get_latents_pca(self, num_PCs=3):
        latents = self.get_latents()
        B, T, N = latents.shape
        latents = latents.reshape(-1, N)
        pca = PCA(n_components=num_PCs)
        latents_pca = pca.fit_transform(latents)
        latents_pca = latents.reshape(B, T, num_PCs)
        return latents_pca, pca

    def compute_FPs(
        self,
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
        if inputs is None:
            _, inputs, _ = self.get_model_input()
        latents = self.get_latents()

        fps = find_fixed_points(
            model=self.wrapper,
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
        is_stable = fps.is_stable
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
        ax.scatter(xstar_pca[:, 0], xstar_pca[:, 1], xstar_pca[:, 2])
        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        plt.show()

    def simulate_neural_data(self):
        self.simulator.simulate_neural_data(
            self.wrapper,
            self.datamodule,
            self.run_name,
            coupled=False,
            seed=0,
        )
