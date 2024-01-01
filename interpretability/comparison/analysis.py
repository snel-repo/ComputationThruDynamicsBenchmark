import os
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA

from DSA import DSA
from interpretability.comparison.fixedpoints import find_fixed_points


class Analysis:
    def __init__(self, run_name, filepath):
        self.run_name = run_name
        self.filepath = filepath

    def load_wrapper(self, filepath):
        # Throw a warning
        return None

    def get_model_output(self):
        return None

    def compute_FPs(self, latents, inputs):
        return None


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
        tt_train_ds = self.datamodule.train_ds
        tt_val_ds = self.datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_targets = torch.vstack((tt_train_ds.tensors[2], tt_val_ds.tensors[2]))
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
            mode="tt",
            n_inits=1024,
            noise_scale=0.01,
            learning_rate=1e-3,
            tol_q=1e-6,
            tol_dq=1e-20,
            tol_unique=1e-1,
            max_iters=50000,
            random_seed=0,
            do_fp_sort=False,
            device="cpu",
            seed=0,
            input_cond=inputs,
        )
        return fps


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
        dt_spiking = torch.vstack((dt_train_ds.tensors[0], dt_val_ds.tensors[0]))
        dt_inputs = torch.vstack((dt_train_ds.tensors[2], dt_val_ds.tensors[2]))
        return dt_spiking, dt_inputs

    def get_model_output(self):
        dt_spiking, dt_inputs = self.get_model_input()
        out_dict = self.model(dt_spiking, dt_inputs)
        return out_dict


class MultiComparator:
    def __init__(self, suffix):
        self.models = []
        self.num_models = 0
        self.suffix = suffix
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{suffix}/"
        )
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

    def load_wrapper(self, analysis):
        self.models.append(analysis)
        self.num_models += 1

    def perform_dsa(self, n_delays, rank, delay_interval, verbose, iters, lr, num_PCs):
        # Compute latent activity from task trained model

        # tt_latents = self.get_tt_latents_pca(num_PCs=num_PCs)
        latents = []
        for i in range(self.num_models):
            latents.append(self.models[i].get_latents_pca(num_PCs=num_PCs))
        similarities = np.zeros((self.num_models, self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                tt_latents_1 = latents[i]
                tt_latents_2 = latents[j]
                dsa = DSA(
                    X=tt_latents_1,
                    Y=tt_latents_2,
                    n_delays=n_delays,
                    rank=rank,
                    delay_interval=delay_interval,
                    verbose=verbose,
                    iters=iters,
                    lr=lr,
                )
                similarities[i, j] = dsa.fit_score()
                print(
                    "Similarity between"
                    f"{self.tt_names[i]} and"
                    f"{self.tt_names[j]}:"
                    f"{similarities[i,j]}"
                )
        print("Final Similarity")
        print(similarities)
        return similarities
