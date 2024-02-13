import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from DSA.stats import dsa_bw_data_splits, dsa_to_id
from interpretability.comparison.analysis.analysis import Analysis
from interpretability.comparison.fixedpoints import find_fixed_points


class Analysis_TT(Analysis):
    def __init__(self, run_name, filepath):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath)
        self.run_hps = None

    def load_wrapper(self, filepath):

        with open(filepath + "model.pkl", "rb") as f:
            self.wrapper = pickle.load(f)
        self.env = self.wrapper.task_env
        self.model = self.wrapper.model
        with open(filepath + "datamodule_train.pkl", "rb") as f:
            self.datamodule = pickle.load(f)
            self.datamodule.prepare_data()
            self.datamodule.setup()

        self.task_name = self.datamodule.data_env.dataset_name
        # if the simulator exists
        if Path(filepath + "simulator.pkl").exists():
            with open(filepath + "simulator.pkl", "rb") as f:
                self.simulator = pickle.load(f)

    def get_model_input(self):
        all_data = self.datamodule.all_data
        tt_ics = torch.Tensor(all_data["ics"])
        tt_inputs = torch.Tensor(all_data["inputs"])
        tt_targets = torch.Tensor(all_data["targets"])
        return tt_ics, tt_inputs, tt_targets

    def get_extra_input(self):
        all_data = self.datamodule.all_data
        tt_extra = torch.Tensor(all_data["extra"])
        return tt_extra

    def get_model_input_noiseless(self):
        all_data = self.datamodule.all_data
        tt_ics = torch.Tensor(all_data["ics"])
        tt_inputs = torch.Tensor(all_data["true_inputs"])
        tt_targets = torch.Tensor(all_data["targets"])
        return tt_ics, tt_inputs, tt_targets

    def get_model_output(self):
        tt_ics, tt_inputs, tt_targets = self.get_model_input()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_model_output_noiseless(self):
        tt_ics, tt_inputs, tt_targets = self.get_model_input_noiseless()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        return out_dict

    def get_latents(self):
        out_dict = self.get_model_output()
        return out_dict["latents"]

    def get_latents_noiseless(self):
        out_dict = self.get_model_output_noiseless()
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
            _, inputs, _ = self.get_model_input_noiseless()
            latents = self.get_latents_noiseless()
        elif inputs is None and not noiseless:
            _, inputs, _ = self.get_model_input()
            latents = self.get_latents()
        else:
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

    def simulate_neural_data(self):
        self.simulator.simulate_neural_data(
            self.wrapper,
            self.datamodule,
            self.run_name,
            coupled=False,
            seed=0,
        )

    def find_DSA_hps(
        self,
        rank_sweep=[10, 20],
        delay_sweep=[1, 5],
    ):
        id_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        splits_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        for i, rank in enumerate(rank_sweep):
            for j, delay in enumerate(delay_sweep):
                print(f"Rank: {rank}, Delay: {delay}")
                id_comp[i, j] = dsa_to_id(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
                splits_comp[i, j] = dsa_bw_data_splits(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(id_comp)
        ax.set_title("ID")
        ax.set_xticks(np.arange(len(delay_sweep)))
        ax.set_yticks(np.arange(len(rank_sweep)))
        ax.set_xticklabels(delay_sweep)
        ax.set_yticklabels(rank_sweep)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig("id_comp.png")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(splits_comp)
        ax2.set_title("Splits")
        ax2.set_xticks(np.arange(len(delay_sweep)))
        ax2.set_yticks(np.arange(len(rank_sweep)))
        ax2.set_xticklabels(delay_sweep)
        ax2.set_yticklabels(rank_sweep)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig("splits_comp.png")
        return id_comp, splits_comp
