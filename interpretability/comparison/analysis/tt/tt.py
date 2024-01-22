import pickle

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

    def compute_FPs(self, inputs):
        # Compute latent activity from task trained model
        latents = self.get_latents()

        fps = find_fixed_points(
            model=self.wrapper,
            state_trajs=latents,
            inputs=inputs,
            n_inits=1024,
            noise_scale=0.01,
            learning_rate=1e-3,
            max_iters=50000,
            device="cpu",
            seed=0,
            compute_jacobians=False,
        )
        return fps
