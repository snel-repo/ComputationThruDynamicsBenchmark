import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# import linear regression from SKLearn
from sklearn.linear_model import LinearRegression

from DSA import DSA
from interpretability.comparison.fixedpoints import find_fixed_points


#
class MultiComparator:
    def __init__(self, suffix):
        self.task_train_wrapper = []
        self.data_train_wrapper = []
        self.env = []
        self.tt_model = []
        self.dt_model = []
        self.simulator = []
        self.tt_datamodule = []
        self.dt_datamodule = []
        self.suffix = suffix
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{suffix}/"
        )
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.latents_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            "interpretability/comparison/latents/"
        )
        self.tt_fps = []
        self.dt_fps = []
        self.tt_names = []
        self.dt_names = []

    def load_task_train_wrapper(self, filepath):
        # if self.task_train_wrapper is  empty, load the first one
        with open(filepath + "model.pkl", "rb") as f:
            wrapper = pickle.load(f)
            self.task_train_wrapper.append(wrapper)
        self.env.append(wrapper.task_env)
        self.tt_model.append(wrapper.model)
        with open(filepath + "datamodule.pkl", "rb") as f:
            tt_datamodule = pickle.load(f)
            tt_datamodule.prepare_data()
            tt_datamodule.setup()
            self.tt_datamodule.append(tt_datamodule)
        with open(filepath + "simulator.pkl", "rb") as f:
            self.simulator.append(pickle.load(f))
        self.tt_names.append(filepath.split("/")[-2])

    def load_data_train_wrapper(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.data_train_wrapper.append(pickle.load(f))
        self.dt_model.append(self.data_train_wrapper)
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.dt_datamodule.append(pickle.load(f))
        self.dt_names.append(filepath.split("/")[-2])

    def get_tt_latents(self):
        num_tt_models = len(self.tt_model)
        tt_latents = []
        for i in range(num_tt_models):
            tt_model = self.tt_model[i]
            tt_datamod = self.tt_datamodule[i]
            tt_model.eval()
            tt_train_ds = tt_datamod.train_ds
            tt_val_ds = tt_datamod.valid_ds

            tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
            tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
            tt_wrapper = self.task_train_wrapper[i]
            tt_model_out, tt_latents_temp, tt_actions = tt_wrapper(tt_ics, tt_inputs)
            tt_latents.append(tt_latents_temp.detach().numpy())

        return tt_latents

    def get_tt_latents_pca(self, num_PCs):
        pca = PCA(n_components=num_PCs)
        num_tt_models = len(self.tt_model)
        tt_latents = self.get_tt_latents()
        tt_latents_pca = []
        for i in range(num_tt_models):
            tt_latents_temp = tt_latents[i]
            num_lats = tt_latents_temp.shape[-1]

            tt_latents_temp = tt_latents_temp.reshape(-1, tt_latents_temp.shape[-1])
            if num_lats < num_PCs:
                # Pad with zeros to get to num_PCs
                tt_latents_temp = np.pad(
                    tt_latents_temp, ((0, 0), (0, num_PCs - num_lats)), "constant"
                )
            tt_latents_pca_temp = pca.fit_transform(tt_latents_temp)
            tt_latents_pca_temp = tt_latents_pca_temp.reshape(
                tt_latents[i].shape[0], tt_latents[i].shape[1], num_PCs
            )
            tt_latents_pca.append(tt_latents_pca_temp)
        return tt_latents_pca

    def perform_dsa(self, n_delays, rank, delay_interval, verbose, iters, lr, num_PCs):
        # Compute latent activity from task trained model

        # tt_latents = self.get_tt_latents_pca(num_PCs=num_PCs)
        tt_latents = self.get_tt_latents()
        similarities = np.zeros((len(self.tt_model), len(self.tt_model)))
        for i in range(len(self.tt_model)):
            for j in range(len(self.tt_model)):
                tt_latents_1 = tt_latents[i]
                tt_latents_2 = tt_latents[j]
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


class Comparisons:
    # TODO
    def __init__(self, suffix):
        self.task_train_wrapper = None
        self.data_train_wrapper = None
        self.env = None
        self.tt_model = None
        self.dt_model = None
        self.simulator = None
        self.tt_datamodule = None
        self.dt_datamodule = None
        self.suffix = suffix
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{suffix}/"
        )
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.latents_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            "interpretability/comparison/latents/"
        )
        self.tt_fps = None
        self.dt_fps = None

    def load_task_train_wrapper(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.task_train_wrapper = pickle.load(f)
        self.env = self.task_train_wrapper.task_env
        self.tt_model = self.task_train_wrapper.model
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.tt_datamodule = pickle.load(f)
            self.tt_datamodule.prepare_data()
            self.tt_datamodule.setup()
        with open(filepath + "simulator.pkl", "rb") as f:
            self.simulator = pickle.load(f)

    def load_data_train_wrapper(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.data_train_wrapper = pickle.load(f)
        self.dt_model = self.data_train_wrapper
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.dt_datamodule = pickle.load(f)

    def getTTLatentActivity(self):
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        _, tt_latents, _ = self.task_train_wrapper(tt_ics, tt_inputs)
        return tt_latents

    def plotLatentActivity(self):
        # Compute latent activity from task trained model
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_outputs = torch.vstack((tt_train_ds.tensors[2], tt_val_ds.tensors[2]))
        tt_model_out, tt_latents, _ = self.task_train_wrapper(tt_ics, tt_inputs)
        dt_train_ds = self.dt_datamodule.train_ds
        dt_val_ds = self.dt_datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4]
        dt_val_inds = dt_val_ds.tensors[4]
        # transform to int
        dt_train_inds = dt_train_inds.type(torch.int64)
        dt_val_inds = dt_val_inds.type(torch.int64)

        tt_latents_val = tt_latents[dt_val_inds]
        tt_model_out_val = tt_model_out[dt_val_inds]
        tt_outputs_val = tt_outputs[dt_val_inds]
        tt_latents_val = tt_latents_val.detach().numpy()
        tt_model_out_val = tt_model_out_val.detach().numpy()
        dt_spikes = dt_val_ds.tensors[0]
        dt_inputs = dt_val_ds.tensors[2]
        dt_log_rates, dt_latents = self.dt_model(dt_spikes, dt_inputs)
        dt_latents = dt_latents.detach().numpy()
        dt_log_rates = dt_log_rates.detach().numpy()

        # Perform PCA on the latents
        Btt, Ttt, Ntt = tt_latents_val.shape
        tt_latents_val = tt_latents_val.reshape(-1, Ntt)
        tt_pca = PCA(n_components=3)
        tt_latents_val = tt_pca.fit_transform(tt_latents_val)

        Bdt, Tdt, Ndt = dt_latents.shape
        dt_latents = dt_latents.reshape(-1, Ndt)
        dt_pca = PCA(n_components=3)
        dt_latents = dt_pca.fit_transform(dt_latents)
        tt_latents_val = tt_latents_val.reshape(Btt, Ttt, 3)
        dt_latents = dt_latents.reshape(Bdt, Tdt, 3)
        n_trials = 5
        fig = plt.figure(figsize=(5 * n_trials, 15))
        for i in range(n_trials):
            ax = fig.add_subplot(3, n_trials, i + 1)
            ax.plot(tt_outputs_val[i, :, 0])
            ax.plot(tt_outputs_val[i, :, 1])
            ax.plot(tt_outputs_val[i, :, 2])
            ax.set_title(f"Trial {i}")
            if i == 0:
                ax.set_ylabel("Output")

            ax = fig.add_subplot(3, n_trials, i + 1 + n_trials)
            ax.plot(tt_latents_val[i, :, 0])
            ax.plot(tt_latents_val[i, :, 1])
            ax.plot(tt_latents_val[i, :, 2])
            if i == 0:
                ax.set_ylabel("TT_Latents (PCs)")

            ax = fig.add_subplot(3, n_trials, i + 1 + 2 * n_trials)
            ax.plot(dt_latents[i, :, 0])
            ax.plot(dt_latents[i, :, 1])
            ax.plot(dt_latents[i, :, 2])
            if i == 0:
                ax.set_ylabel("DT_Latents (PCs)")

        path2 = self.suffix + "_latent_activity"
        plt.savefig(self.plot_path + path2 + ".png")
        plt.savefig(self.plot_path + path2 + ".pdf")

        dt_latents = dt_latents.reshape(-1, 3)
        tt_latents_val = tt_latents_val.reshape(-1, 3)

        # Affine transform the latents
        lr = LinearRegression().fit(dt_latents, tt_latents_val)
        dt_latents = lr.predict(dt_latents)

        tt_latents_val = tt_latents_val.reshape(Btt, Ttt, 3)
        dt_latents = dt_latents.reshape(Bdt, Tdt, 3)

        # Plot the outputs of the env adn the latents from each model for a few trials
        n_trials = 1
        fig = plt.figure(figsize=(10 * n_trials, 10))
        for i in range(n_trials):
            ax = fig.add_subplot(2, n_trials, i + 1)
            ax.plot(tt_model_out_val[i, :, 0])
            ax.plot(tt_model_out_val[i, :, 1])
            ax.plot(tt_model_out_val[i, :, 2])

            ax = fig.add_subplot(2, n_trials, i + 1 + n_trials)
            ax.plot(tt_outputs_val[i, :, 0])
            ax.plot(tt_outputs_val[i, :, 1])
            ax.plot(tt_outputs_val[i, :, 2])

        path2 = self.suffix + "_outputs"
        plt.savefig(self.plot_path + path2 + ".png")
        plt.savefig(self.plot_path + path2 + ".pdf")

        n_trials = 5
        fig = plt.figure(figsize=(5 * n_trials, 15))
        for i in range(n_trials):
            ax = fig.add_subplot(3, n_trials, i + 1)
            ax.plot(tt_outputs_val[i, :, 0])
            ax.plot(tt_outputs_val[i, :, 1])
            ax.plot(tt_outputs_val[i, :, 2])
            if i == 0:
                ax.set_ylabel("Output")

            ax = fig.add_subplot(3, n_trials, i + 1 + n_trials)
            ax.plot(tt_latents_val[i, :, 0])
            ax.plot(tt_latents_val[i, :, 1])
            ax.plot(tt_latents_val[i, :, 2])
            if i == 0:
                ax.set_ylabel("TT_Latents (PCs)")

            ax = fig.add_subplot(3, n_trials, i + 1 + 2 * n_trials)
            ax.plot(dt_latents[i, :, 0])
            ax.plot(dt_latents[i, :, 1])
            ax.plot(dt_latents[i, :, 2])
            if i == 0:
                ax.set_ylabel("DT_Latents (PCs)")
        path2 = self.suffix + "_latent_activity"
        plt.savefig(self.plot_path + path2 + ".png")
        plt.savefig(self.plot_path + path2 + ".pdf")

        # Plot 3D trajectories of latents
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(n_trials):
            ax.plot(
                tt_latents_val[i, :, 0],
                tt_latents_val[i, :, 1],
                tt_latents_val[i, :, 2],
            )

        if self.tt_fps is not None:
            tt_fps = self.tt_fps.xstar
            tt_fps = tt_pca.transform(tt_fps)
            stable = self.tt_fps.is_stable
            ax.scatter(
                tt_fps[stable, 0],
                tt_fps[stable, 1],
                tt_fps[stable, 2],
                color="k",
                s=100,
            )
            ax.scatter(
                tt_fps[~stable, 0],
                tt_fps[~stable, 1],
                tt_fps[~stable, 2],
                color="r",
                s=50,
            )
        ax.set_title("Task trained latents")
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(n_trials):
            ax.plot(dt_latents[i, :, 0], dt_latents[i, :, 1], dt_latents[i, :, 2])
        if self.dt_fps is not None:
            dt_fps = self.dt_fps.xstar
            dt_fps = dt_pca.transform(dt_fps)
            dt_fps = lr.predict(dt_fps)
            stable = self.dt_fps.is_stable
            ax.scatter(
                dt_fps[stable, 0],
                dt_fps[stable, 1],
                dt_fps[stable, 2],
                color="k",
                s=100,
            )
            ax.scatter(
                dt_fps[~stable, 0],
                dt_fps[~stable, 1],
                dt_fps[~stable, 2],
                color="r",
                s=50,
            )
        ax.set_title("Data trained latents")
        plt.savefig(self.plot_path + self.suffix + "_latent_trajectories_fps.png")
        plt.savefig(self.plot_path + self.suffix + "_latent_trajectories_fps.pdf")

        # Plot spiking for data simulation
        dt_spikes = dt_spikes.detach().numpy()
        dt_datagen_act = dt_val_ds.tensors[5]
        dt_datagen_lats = dt_val_ds.tensors[3]

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        trial_num = 1
        ax1.imshow(dt_spikes[trial_num, :, :].T, aspect="auto", interpolation=None)
        plt.savefig(self.plot_path + self.suffix + "_spikes.png")
        plt.savefig(self.plot_path + self.suffix + "_spikes.svg")

        fig = plt.figure(figsize=(5, 5))
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.imshow(dt_datagen_act[trial_num, :, :].T, aspect="auto", interpolation=None)
        plt.savefig(self.plot_path + self.suffix + "_act.png")
        plt.savefig(self.plot_path + self.suffix + "_act.pdf")

        fig = plt.figure(figsize=(5, 5))
        ax3 = fig.add_subplot(1, 1, 1)
        ax3.imshow(
            dt_datagen_lats[trial_num, :, :].T, aspect="auto", interpolation=None
        )
        plt.savefig(self.plot_path + self.suffix + "_datagen_lats.png")
        plt.savefig(self.plot_path + self.suffix + "_datagen_lats.pdf")

        fig = plt.figure(figsize=(5, 5))
        ax3 = fig.add_subplot(1, 1, 1)
        ax3.imshow(dt_log_rates[trial_num, :, :].T, aspect="auto", interpolation=None)
        plt.savefig(self.plot_path + self.suffix + "_dt_logrates.png")
        plt.savefig(self.plot_path + self.suffix + "_dt_logrates.pdf")

        fig = plt.figure(figsize=(5, 5))
        ax3 = fig.add_subplot(1, 1, 1)
        ax3.imshow(
            np.exp(dt_log_rates[trial_num, :, :].T), aspect="auto", interpolation=None
        )
        plt.savefig(self.plot_path + self.suffix + "_dt_rates.png")
        plt.savefig(self.plot_path + self.suffix + "_dt_rates.pdf")

    @staticmethod
    def compareLatentActivityPath(dtLatentsPath, ttLatentsPath):

        with open(dtLatentsPath, "rb") as f:
            dt_dict = pickle.load(f)
        with open(ttLatentsPath, "rb") as f:
            tt_dict = pickle.load(f)

        tt_latents = tt_dict["tt_latents"]
        dt_latents = dt_dict["dt_latents"]

        tt_latents_flat = tt_latents.reshape(-1, tt_latents.shape[-1])
        dt_latents_flat = dt_latents.reshape(-1, dt_latents.shape[-1])

        pca_tt = PCA(n_components=3)
        tt_lats_pca = pca_tt.fit_transform(tt_latents_flat)
        pca_dt = PCA(n_components=3)
        dt_lats_pca = pca_dt.fit_transform(dt_latents_flat)

        # Perform PCA on the latents
        Btt, Ttt, Ntt = tt_latents.shape
        tt_latents_flat = tt_latents.reshape(-1, Ntt)
        Bdt, Tdt, Ndt = dt_latents.shape
        dt_latents_flat = dt_latents.reshape(-1, Ndt)

        print("Fitting all latent dimensions")
        # Fit linear regression between the latents
        lr_getSys = LinearRegression().fit(dt_latents_flat, tt_latents_flat)
        r2_getSys = lr_getSys.score(dt_latents_flat, tt_latents_flat)
        lr_noExtra = LinearRegression().fit(tt_latents_flat, dt_latents_flat)
        r2_noExtra = lr_noExtra.score(tt_latents_flat, dt_latents_flat)

        print(f"Rate R2: {r2_getSys}")
        print(f"State R2: {r2_noExtra}")

        print("Fitting top 3 PCs only")
        lr_getSysPCA = LinearRegression().fit(dt_lats_pca, tt_lats_pca)
        r2_getSysPCA = lr_getSysPCA.score(dt_lats_pca, tt_lats_pca)
        lr_noExtraPCA = LinearRegression().fit(tt_lats_pca, dt_lats_pca)
        r2_noExtraPCA = lr_noExtraPCA.score(tt_lats_pca, dt_lats_pca)

        print(f"Rate R2 PCA: {r2_getSysPCA}")
        print(f"State R2 PCA: {r2_noExtraPCA}")

        dt_lats_pca = lr_getSysPCA.predict(dt_lats_pca)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        dt_lats_pca = dt_lats_pca.reshape(Bdt, Tdt, 3)
        tt_lats_pca = tt_lats_pca.reshape(Btt, Ttt, 3)
        for i in range(10):
            ax.plot(dt_lats_pca[i, :, 0], dt_lats_pca[i, :, 1], dt_lats_pca[i, :, 2])
        ax.set_title("Data trained latents")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(10):
            ax.plot(tt_lats_pca[i, :, 0], tt_lats_pca[i, :, 1], tt_lats_pca[i, :, 2])
        ax.set_title("Task trained latents")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        return_dict = {
            "r2_getSys": r2_getSys,
            "r2_noExtra": r2_noExtra,
            "r2_getSysPCA": r2_getSysPCA,
            "r2_noExtraPCA": r2_noExtraPCA,
        }

        # if "tt_fps" in tt_dict.keys() and "dt_fps" in dt_dict.keys():
        #     tt_fps = tt_dict["tt_fps"]
        #     dt_fps = dt_dict["dt_fps"]

        return return_dict

    def plotHandKinematics(self):
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_targets = torch.vstack((tt_train_ds.tensors[2], tt_val_ds.tensors[2]))
        controlled, tt_latents, actions = self.task_train_wrapper(
            tt_ics, tt_inputs, tt_targets
        )
        controlled = controlled.detach().numpy()
        actions = actions.detach().numpy()
        for i in range(10):
            fig = plt.figure()
            ax = fig.add_subplot(3, 1, 1)
            # Make a yellow square at the target position
            ax.plot(tt_targets[i, :, 0], tt_targets[i, :, 1], color="y", marker="s")
            ax.plot(controlled[i, :, 0], controlled[i, :, 1])

            ax.set_title(f"Trial {i}")
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_aspect("equal", "box")

            ax = fig.add_subplot(3, 1, 2)
            ax.plot(np.diff(controlled[i, :, 0]), color="r", label="xVel")
            ax.plot(np.diff(controlled[i, :, 1]), color="b", label="yVel")
            ax.legend()

            ax = fig.add_subplot(3, 1, 3)
            ax.plot(actions[i, :, 0], color="r", label="M1")
            ax.plot(actions[i, :, 1], color="b", label="M2")
            ax.plot(actions[i, :, 2], color="g", label="M3")
            ax.plot(actions[i, :, 3], color="k", label="M4")
            # Legend aligned to right
            ax.legend(loc="center right", bbox_to_anchor=(1, 0.5))

            plt.savefig(self.plot_path + self.suffix + f"_trial_{i}.png")

    def saveComparisonDict(self):
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_model_out, tt_latents, _ = self.task_train_wrapper(tt_ics, tt_inputs)
        dt_train_ds = self.dt_datamodule.train_ds
        dt_val_ds = self.dt_datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4]
        dt_val_inds = dt_val_ds.tensors[4]
        # transform to int
        dt_train_inds = dt_train_inds.type(torch.int64)
        dt_val_inds = dt_val_inds.type(torch.int64)

        tt_latents_val = tt_latents[dt_val_inds]
        tt_model_out_val = tt_model_out[dt_val_inds]
        tt_latents_val = tt_latents_val.detach().numpy()
        tt_model_out_val = tt_model_out_val.detach().numpy()
        dt_spikes = dt_val_ds.tensors[0]
        dt_inputs = dt_val_ds.tensors[2]
        dt_log_rates, dt_latents = self.dt_model(dt_spikes, dt_inputs)
        dt_latents = dt_latents.detach().numpy()
        dt_log_rates = dt_log_rates.detach().numpy()

        tt_dict = {
            "tt_latents": tt_latents_val,
        }
        dt_dict = {
            "dt_latents": dt_latents,
        }

        if self.tt_fps is not None:
            fp_list = []
            for i in range(len(self.tt_fps)):
                tt_fps = self.tt_fps[i].xstar
                tt_stable = self.tt_fps[i].is_stable
                tt_q = self.tt_fps[i].qstar
                tt_dq = self.tt_fps[i].dq
                tt_eigs = self.tt_fps[i].eigval_J_xstar
                fp_dict = {
                    "tt_fps": tt_fps,
                    "tt_stable": tt_stable,
                    "tt_q": tt_q,
                    "tt_dq": tt_dq,
                    "tt_eigs": tt_eigs,
                }
                fp_list.append(fp_dict)
            tt_dict["tt_fps"] = fp_list

        if self.dt_fps is not None:
            fp_list = []
            for i in range(len(self.dt_fps)):
                dt_fps = self.dt_fps[i].xstar
                dt_stable = self.dt_fps[i].is_stable
                dt_q = self.dt_fps[i].qstar
                dt_dq = self.dt_fps[i].dq
                dt_eigs = self.dt_fps[i].eigval_J_xstar
                fp_dict = {
                    "dt_fps": dt_fps,
                    "dt_stable": dt_stable,
                    "dt_q": dt_q,
                    "dt_dq": dt_dq,
                    "dt_eigs": dt_eigs,
                }
                fp_list.append(fp_dict)
            dt_dict["dt_fps"] = fp_list

        # Save latents as pickle
        with open(self.latents_path + self.suffix + "_tt_latents.pkl", "wb") as f:
            pickle.dump(tt_dict, f)
        with open(self.latents_path + self.suffix + "_dt_latents.pkl", "wb") as f:
            pickle.dump(dt_dict, f)

    def compareLatentActivity(self):
        # Compute latent activity from task trained model
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_model_out, tt_latents, tt_actions = self.task_train_wrapper(
            tt_ics, tt_inputs
        )
        dt_train_ds = self.dt_datamodule.train_ds
        dt_val_ds = self.dt_datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4]
        dt_val_inds = dt_val_ds.tensors[4]
        # transform to int
        dt_train_inds = dt_train_inds.type(torch.int64)
        dt_val_inds = dt_val_inds.type(torch.int64)

        tt_latents_val = tt_latents[dt_val_inds]
        tt_model_out_val = tt_model_out[dt_val_inds]
        tt_latents_val = tt_latents_val.detach().numpy()
        tt_model_out_val = tt_model_out_val.detach().numpy()
        dt_spikes = dt_val_ds.tensors[0]
        dt_inputs = dt_val_ds.tensors[2]
        dt_log_rates, dt_latents = self.dt_model(dt_spikes, dt_inputs)
        dt_latents = dt_latents.detach().numpy()
        dt_log_rates = dt_log_rates.detach().numpy()

        # Perform PCA on the latents
        Btt, Ttt, Ntt = tt_latents_val.shape
        tt_latents_flat = tt_latents_val.reshape(-1, Ntt)
        Bdt, Tdt, Ndt = dt_latents.shape
        dt_latents_flat = dt_latents.reshape(-1, Ndt)

        pca_tt = PCA(n_components=3)
        tt_lats_pca = pca_tt.fit_transform(tt_latents_flat)
        pca_dt = PCA(n_components=3)
        dt_lats_pca = pca_dt.fit_transform(dt_latents_flat)

        # Fit linear regression between the latents
        lr_getSys = LinearRegression().fit(dt_latents_flat, tt_latents_flat)
        r2_getSys = lr_getSys.score(dt_latents_flat, tt_latents_flat)
        lr_noExtra = LinearRegression().fit(tt_latents_flat, dt_latents_flat)
        r2_noExtra = lr_noExtra.score(tt_latents_flat, dt_latents_flat)
        print(f"{self.suffix}")
        print(f"Rate R2: {r2_getSys}")
        print(f"State R2: {r2_noExtra}")

        lr_getSysPCA = LinearRegression().fit(dt_lats_pca, tt_lats_pca)
        r2_getSysPCA = lr_getSysPCA.score(dt_lats_pca, tt_lats_pca)
        lr_noExtraPCA = LinearRegression().fit(tt_lats_pca, dt_lats_pca)
        r2_noExtraPCA = lr_noExtraPCA.score(tt_lats_pca, dt_lats_pca)
        print(f"{self.suffix} Top 3 PCs only")
        print(f"Rate R2 PCA: {r2_getSysPCA}")
        print(f"State R2 PCA: {r2_noExtraPCA}")
        dt_lats_pca = lr_getSysPCA.predict(dt_lats_pca)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        dt_lats_pca = dt_lats_pca.reshape(Bdt, Tdt, 3)
        tt_lats_pca = tt_lats_pca.reshape(Btt, Ttt, 3)
        for i in range(10):
            ax.plot(dt_lats_pca[i, :, 0], dt_lats_pca[i, :, 1], dt_lats_pca[i, :, 2])
        ax.set_title("Data trained latents")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(10):
            ax.plot(tt_lats_pca[i, :, 0], tt_lats_pca[i, :, 1], tt_lats_pca[i, :, 2])
        ax.set_title("Task trained latents")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(self.plot_path + self.suffix + "_latent_trajectories_R2.png")
        plt.savefig(self.plot_path + self.suffix + "_latent_trajectories_R2.pdf")

        return_dict = {
            "r2_getSys": r2_getSys,
            "r2_noExtra": r2_noExtra,
            "r2_getSysPCA": r2_getSysPCA,
            "r2_noExtraPCA": r2_noExtraPCA,
        }
        return return_dict

    def plotTrial(self, trial_num=0):
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_outputs = torch.vstack((tt_train_ds.tensors[2], tt_val_ds.tensors[2]))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(tt_inputs[trial_num, :, 0], label="x")
        ax.plot(tt_inputs[trial_num, :, 1], label="y")
        ax.plot(tt_inputs[trial_num, :, 2], label="z")
        ax.set_title("Inputs")
        ax.legend()
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(tt_outputs[trial_num, :, 0])
        ax.plot(tt_outputs[trial_num, :, 1])
        ax.plot(tt_outputs[trial_num, :, 2])
        ax.set_title("Outputs")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}.png")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}.pdf")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot(
            tt_outputs[trial_num, :, 0],
            tt_outputs[trial_num, :, 1],
            tt_outputs[trial_num, :, 2],
        )
        ax.set_title("State")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}_state.png")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}_state.pdf")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        for i in range(10):
            ax.plot(tt_outputs[i, :, 0], tt_outputs[i, :, 1], tt_outputs[i, :, 2])
        ax.set_title("State")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}_state_10.png")
        plt.savefig(f"{self.plot_path}{self.suffix}_trial{trial_num}_state_10.pdf")

    def compute_TT_FPs_MultiTask(self, task_to_analyze, phase, plot_fps=False):
        # Set model to Eval mode
        self.tt_model.eval()

        # Get the data from the datamodule
        data_dict = self.tt_datamodule.all_data
        ics = data_dict["ics"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        true_inputs = data_dict["true_inputs"]
        readout = self.task_train_wrapper.model.readout

        # find indices where task_to_analyze is the task
        task_inds = [i for i, task in enumerate(task_names) if task == task_to_analyze]

        # Get the inputs, ics, and phase_dict for the task
        task_inputs = torch.Tensor(true_inputs[task_inds])
        task_ics = torch.Tensor(ics[task_inds])
        task_phase_dict = [
            dict1 for i, dict1 in enumerate(phase_dict) if i in task_inds
        ]

        # Pass data through the model
        tt_outputs, tt_latents, _ = self.task_train_wrapper(task_ics, task_inputs)

        # Get the latents, inputs, and outputs for the phase of interest
        task_ics_phase = []
        task_inputs_phase = []
        task_outputs_phase = []
        tt_latents_phase = []
        for i in range(len(task_phase_dict)):
            phase_edges = task_phase_dict[i][phase]
            task_ics_phase.append(task_ics[i][phase_edges[0] : phase_edges[1]])
            task_inputs_phase.append(task_inputs[i][phase_edges[0] : phase_edges[1]])
            task_outputs_phase.append(tt_outputs[i][phase_edges[0] : phase_edges[1]])
            tt_latents_phase.append(tt_latents[i][phase_edges[0] : phase_edges[1]])

        # Stack along first dimension
        task_ics_phase = torch.vstack(task_ics_phase)
        task_inputs_phase = torch.vstack(task_inputs_phase)
        tt_outputs_phase = torch.vstack(task_outputs_phase)
        tt_latents_phase = torch.vstack(tt_latents_phase)

        # Compute the fixed points
        tt_fps = find_fixed_points(
            model=self.task_train_wrapper,
            state_trajs=tt_latents_phase,
            inputs=task_inputs_phase,
            n_inits=1024,
            noise_scale=0.1,
            learning_rate=5e-3,
            max_iters=10000,
            device="cpu",
            seed=0,
            compute_jacobians=False,
        )
        self.tt_fps = tt_fps
        q_vals = tt_fps.qstar

        # Plot histogram of q values
        if plot_fps:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(q_vals, bins=100)
            ax.set_title("Histogram of Q values")
            plt.savefig(self.plot_path + self.suffix + "_q_hist.png")

        # PCA the latents
        tt_fp_loc = tt_fps.xstar
        tt_latents = tt_latents_phase.detach().numpy()
        tt_outputs = tt_outputs_phase.detach().numpy()
        tt_latents_flat = tt_latents.reshape(-1, tt_latents.shape[-1])
        pca_tt = PCA(n_components=3)
        tt_lats_pca = pca_tt.fit_transform(tt_latents_flat)
        tt_fp_loc_pca = pca_tt.transform(tt_fp_loc)
        tt_lats_pca = tt_lats_pca.reshape(tt_latents.shape[0], -1, 3)

        # Get the projection onto the readout for the fps
        tt_fp_readout = readout(torch.Tensor(tt_fp_loc))
        if plot_fps:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.scatter(
                tt_fp_loc_pca[:, 0],
                tt_fp_loc_pca[:, 1],
                tt_fp_readout[:, 2],
                # set color by q value
                c=q_vals,
                cmap="viridis",
            )
            ax.set_title("Task trained latents")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            maxXY = np.max(np.abs(tt_fp_loc_pca[:, :2]))
            ax.set_xlim([-1.1 * maxXY, 1.1 * maxXY])
            ax.set_ylim([-1.1 * maxXY, 1.1 * maxXY])
            ax.set_zlim([-1.1, 1.1])
            plt.savefig(
                self.plot_path
                + self.suffix
                + f"_{task_to_analyze}_{phase}_latent_trajectories_fps.png"
            )

        return tt_fps, tt_fp_readout

    def computeFPs(self, input_conds=None):
        # Compute latent activity from task trained model
        self.tt_model.eval()
        train_datamodule = self.tt_datamodule.train_dataloader()
        train_tt_latents = []
        for batch in train_datamodule:
            ics, inputs, _, _ = batch
            # Pass data through the model
            _, tt_latents, _ = self.task_train_wrapper(ics, inputs)
            train_tt_latents.append(tt_latents)
        # Stack along first dimension
        train_tt_latents = torch.vstack(train_tt_latents)
        # Compute latent activity from data trained model
        self.dt_model.eval()
        train_datamodule = self.dt_datamodule.train_dataloader()
        train_dt_latents = []
        for batch in train_datamodule:
            spikes, _, inputs, *_ = batch
            # Pass data through the model
            _, dt_latents = self.dt_model(spikes, inputs)
            train_dt_latents.append(dt_latents)
        train_dt_latents = torch.vstack(train_dt_latents)
        if input_conds is None:
            input_conds = [torch.zeros((1, inputs.shape[-1]))]

        for _, input_cond in enumerate(input_conds):
            tt_fps = find_fixed_points(
                model=self.task_train_wrapper,
                state_trajs=train_tt_latents,
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
                input_cond=input_cond,
            )
            maxQ = tt_fps.qstar.max()
            print(f"Max Q for included TT-FPs: {maxQ}")

            dt_fps = find_fixed_points(
                model=self.data_train_wrapper,
                state_trajs=train_dt_latents,
                mode="dt",
                n_inits=1024,
                noise_scale=0.01,
                learning_rate=1e-3,
                tol_q=maxQ,
                tol_dq=1e-20,
                tol_unique=1e-1,
                max_iters=50000,
                random_seed=0,
                do_fp_sort=False,
                device="cpu",
                seed=0,
                input_cond=input_cond,
            )

        self.tt_fps = tt_fps
        self.dt_fps = dt_fps
        return tt_fps, dt_fps

    def compareFPs(self):
        tt_fps = self.tt_fps
        dt_fps = self.dt_fps
        tt_fp_loc = tt_fps.xstar
        dt_fp_loc = dt_fps.xstar

        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_model_out, tt_latents, tt_actions = self.task_train_wrapper(
            tt_ics, tt_inputs
        )
        dt_train_ds = self.dt_datamodule.train_ds
        dt_val_ds = self.dt_datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4]
        dt_val_inds = dt_val_ds.tensors[4]
        # transform to int
        dt_train_inds = dt_train_inds.type(torch.int64)
        dt_val_inds = dt_val_inds.type(torch.int64)

        tt_latents_val = tt_latents[dt_val_inds]
        tt_model_out_val = tt_model_out[dt_val_inds]
        tt_latents_val = tt_latents_val.detach().numpy()
        tt_model_out_val = tt_model_out_val.detach().numpy()
        dt_spikes = dt_val_ds.tensors[0]
        dt_inputs = dt_val_ds.tensors[2]
        dt_log_rates, dt_latents = self.dt_model(dt_spikes, dt_inputs)
        dt_latents = dt_latents.detach().numpy()
        dt_log_rates = dt_log_rates.detach().numpy()

        tt_lats_flat = tt_latents_val.reshape(-1, tt_latents_val.shape[-1])
        dt_lats_flat = dt_latents.reshape(-1, dt_latents.shape[-1])

        tt2dt = LinearRegression().fit(tt_lats_flat, dt_lats_flat)
        # dt2tt = LinearRegression().fit(dt_lats_flat, tt_lats_flat)

        tt_fp_loc = tt_fps.xstar
        tt_fp_loc_in_dt = tt2dt.predict(tt_fp_loc)

        dt_fp_loc = dt_fps.xstar

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        ax.scatter(
            dt_fp_loc[:, 0],
            dt_fp_loc[:, 1],
            dt_fp_loc[:, 2],
            color="k",
            s=100,
            label="dt",
        )
        ax.scatter(
            tt_fp_loc_in_dt[:, 0],
            tt_fp_loc_in_dt[:, 1],
            tt_fp_loc_in_dt[:, 2],
            color="b",
            s=100,
            label="tt in dt",
        )

    def compute_data_for_levelset(self):
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_model_out, tt_latents, tt_actions = self.task_train_wrapper(
            tt_ics, tt_inputs
        )
        dt_train_ds = self.dt_datamodule.train_ds
        dt_val_ds = self.dt_datamodule.valid_ds
        dt_train_inds = dt_train_ds.tensors[4]
        dt_val_inds = dt_val_ds.tensors[4]
        # transform to int
        dt_train_inds = dt_train_inds.type(torch.int64)
        dt_val_inds = dt_val_inds.type(torch.int64)

        tt_inputs_train = tt_inputs[dt_train_inds]
        tt_inputs_val = tt_inputs[dt_val_inds]

        tt_latents_train = tt_latents[dt_train_inds]
        tt_latents_val = tt_latents[dt_val_inds]

        tt_latents = torch.vstack((tt_latents_train, tt_latents_val))
        tt_latents = tt_latents.detach().numpy()

        dt_spikes_val = dt_val_ds.tensors[0]
        dt_inputs_val = dt_val_ds.tensors[2]
        dt_spikes_train = dt_train_ds.tensors[0]
        dt_inputs_train = dt_train_ds.tensors[2]

        dt_spikes = torch.vstack((dt_spikes_train, dt_spikes_val))
        dt_inputs = torch.vstack((dt_inputs_train, dt_inputs_val))

        dt_log_rates, dt_latents = self.dt_model(dt_spikes, dt_inputs)
        tt_latents_flat = tt_latents.reshape(-1, tt_latents.shape[-1])
        dt_latents_flat = dt_latents.reshape(-1, dt_latents.shape[-1]).detach().numpy()
        lr_tt2dt = LinearRegression().fit(tt_latents_flat, dt_latents_flat)
        tt_latents_flat = lr_tt2dt.predict(tt_latents_flat)

        tt_latents = tt_latents_flat.reshape(
            tt_latents.shape[0], tt_latents.shape[1], -1
        )
        dt_latents_diff = torch.diff(dt_latents, dim=1)
        dt_inputs = dt_inputs[:, :-1, :]
        dt_latents = dt_latents[:, :-1, :]
        tt_latents = torch.tensor(tt_latents)
        tt_latents_diff = torch.diff(tt_latents, dim=1)

        tt_inputs = torch.vstack((tt_inputs_train, tt_inputs_val))
        tt_inputs = tt_inputs[:, :-1, :]
        tt_latents = tt_latents[:, :-1, :]

        # Get the L2 norm of the latents diff
        q_tt = torch.norm(tt_latents_diff, dim=-1)
        q_dt = torch.norm(dt_latents_diff, dim=-1)
        tt_latents_flat = tt_latents.reshape(-1, tt_latents.shape[-1])
        dt_latents_flat = dt_latents.reshape(-1, dt_latents.shape[-1])
        tt_inputs_flat = tt_inputs.reshape(-1, tt_inputs.shape[-1])
        dt_inputs_flat = dt_inputs.reshape(-1, dt_inputs.shape[-1])
        q_tt_flat = q_tt.reshape(-1, 1)
        q_dt_flat = q_dt.reshape(-1, 1)

        sorted1, tt_sort_inds = torch.sort(q_tt_flat, dim=0)
        sorted2, dt_sort_inds = torch.sort(q_dt_flat, dim=0)
        tt_sort_inds.squeeze_()
        dt_sort_inds.squeeze_()

        tt_datablock = torch.hstack(
            (
                tt_latents_flat[tt_sort_inds, :],
                tt_inputs_flat[tt_sort_inds, :],
                q_tt_flat[tt_sort_inds, :],
            )
        )
        dt_datablock = torch.hstack(
            (
                dt_latents_flat[dt_sort_inds],
                dt_inputs_flat[dt_sort_inds],
                q_dt_flat[dt_sort_inds],
            )
        )
        path1 = "/snel/share/runs/fpfinding/levelset_data/"
        # Save latents as pickle
        tt_savepath = path1 + self.suffix + "_tt_datablock.pkl"
        dt_savepath = path1 + self.suffix + "_dt_datablock.pkl"
        torch.save(tt_datablock, tt_savepath)
        torch.save(dt_datablock, dt_savepath)

        # Save as h5
        tt_savepath = path1 + self.suffix + "_tt_datablock.h5"
        dt_savepath = path1 + self.suffix + "_dt_datablock.h5"
        with h5py.File(tt_savepath, "w") as f:
            f.create_dataset("tt_datablock", data=tt_datablock.detach().numpy())
        with h5py.File(dt_savepath, "w") as f:
            f.create_dataset("dt_datablock", data=dt_datablock.detach().numpy())
        return tt_datablock, dt_datablock

    def compute_FPs_MultiTask(self):
        # Compute latent activity from task trained model

        self.tt_model.eval()

        data_dict = self.tt_datamodule.all_data
        # inputs = data_dict["inputs"]
        ics = data_dict["ics"]
        # targets = data_dict["targets"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        true_inputs = data_dict["true_inputs"]

        tt_latents = []
        tt_outputs = []
        task_to_analyze = "MemoryPro"
        # find indices where task_to_analyze is the task
        task_inds = [i for i, task in enumerate(task_names) if task == task_to_analyze]

        task_inputs = torch.Tensor(true_inputs[task_inds])
        task_ics = torch.Tensor(ics[task_inds])
        task_phase_dict = [
            dict1 for i, dict1 in enumerate(phase_dict) if i in task_inds
        ]

        tt_outputs, tt_latents, _ = self.task_train_wrapper(task_ics, task_inputs)
        readout = self.task_train_wrapper.model.readout
        phases = task_phase_dict[0].keys()
        phase_for_pca = "mem1"

        phase_lats = {}
        phase_outs = {}
        phase_outs_lagged = {}
        phase_lats_lagged = {}
        phase_inputs = {}
        phase_lats_tensor = {}
        phase_outs_tensor = {}
        phase_inputs_tensor = {}

        for phase in phases:
            phase_lats[phase] = []
            phase_outs[phase] = []
            phase_inputs[phase] = []
            phase_lats_lagged[phase] = []
            phase_outs_lagged[phase] = []
            for i, phase_dict in enumerate(task_phase_dict):
                phase_start = phase_dict[phase][0]
                phase_end = phase_dict[phase][1]
                phase_lats[phase].append(tt_latents[i, phase_start:phase_end, :])
                phase_inputs[phase].append(task_inputs[i, phase_start:phase_end, :])
                phase_outs[phase].append(tt_outputs[i, phase_start:phase_end, :])
                if phase_start > 0:
                    phase_lats_lagged[phase].append(
                        tt_latents[i, phase_start - 2 : phase_end, :]
                    )
                    phase_outs_lagged[phase].append(
                        tt_outputs[i, phase_start - 2 : phase_end, :]
                    )
                else:
                    phase_lats_lagged[phase].append(
                        tt_latents[i, phase_start:phase_end, :]
                    )
                    phase_outs_lagged[phase].append(
                        tt_outputs[i, phase_start:phase_end, :]
                    )

            # Stack along first dimension and compute FP
            phase_lats_tensor[phase] = torch.vstack(phase_lats[phase])
            phase_inputs_tensor[phase] = torch.vstack(phase_inputs[phase])
            phase_outs_tensor[phase] = torch.vstack(phase_outs[phase])

        pca_phase = PCA(n_components=2)
        pca_phase.fit(phase_lats_tensor[phase_for_pca].detach().numpy())

        # Panel 1: FP find during Context and Memory phases
        panel1_lats = torch.vstack(
            (phase_lats_tensor["context"], phase_lats_tensor["mem1"])
        )
        panel1_inputs = torch.vstack(
            (phase_inputs_tensor["context"], phase_inputs_tensor["mem1"])
        )

        panel1_fps = find_fixed_points(
            model=self.task_train_wrapper,
            state_trajs=panel1_lats,
            mode="tt",
            n_inits=4096,
            noise_scale=0.1,
            learning_rate=1e-2,
            tol_q=5e-6,
            tol_dq=1e-20,
            tol_unique=1e-2,
            max_iters=10000,
            random_seed=0,
            do_fp_sort=False,
            device="cpu",
            seed=0,
            inputs=panel1_inputs,
            compute_jacobians=False,
            use_percent=True,
        )

        panel2_fps = find_fixed_points(
            model=self.task_train_wrapper,
            state_trajs=phase_lats_tensor["stim1"],
            mode="tt",
            n_inits=4096,
            noise_scale=0.1,
            learning_rate=1e-2,
            tol_q=8e-6,
            tol_dq=1e-20,
            tol_unique=1e-2,
            max_iters=10000,
            random_seed=0,
            do_fp_sort=False,
            device="cpu",
            seed=0,
            inputs=phase_inputs_tensor["stim1"],
            compute_jacobians=False,
            use_percent=True,
        )

        panel4_fps = find_fixed_points(
            model=self.task_train_wrapper,
            state_trajs=phase_lats_tensor["response"],
            mode="tt",
            n_inits=4096,
            noise_scale=0.1,
            learning_rate=1e-2,
            tol_q=1e-4,
            tol_dq=1e-20,
            tol_unique=1e-2,
            max_iters=10000,
            random_seed=0,
            do_fp_sort=False,
            device="cpu",
            seed=0,
            inputs=phase_inputs_tensor["response"],
            compute_jacobians=False,
            use_percent=True,
        )

        # lat_pca_context = pca_phase.transform(phase_lats_tensor["context"])

        # Assuming tt_fps.xstar is a numpy array,
        # if not you may need to adjust this part
        fps_xstar_1 = panel1_fps.xstar
        readout_proj_1 = readout(torch.Tensor(fps_xstar_1))
        x_output_1 = readout_proj_1[:, 1].detach().numpy()

        fps_xstar_2 = panel2_fps.xstar
        readout_proj_2 = readout(torch.Tensor(fps_xstar_2))
        x_output_2 = readout_proj_2[:, 1].detach().numpy()

        x_output_3 = x_output_1

        fps_xstar_4 = panel4_fps.xstar
        readout_proj_4 = readout(torch.Tensor(fps_xstar_4))
        x_output_4 = readout_proj_4[:, 1].detach().numpy()

        # Apply PCA
        fps_xstar_pca_1 = pca_phase.transform(fps_xstar_1)
        fps_xstar_pca_1 = np.hstack((fps_xstar_pca_1, x_output_1[:, None]))
        fps_xstar_pca_2 = pca_phase.transform(fps_xstar_2)
        fps_xstar_pca_2 = np.hstack((fps_xstar_pca_2, x_output_2[:, None]))
        fps_xstar_pca_3 = pca_phase.transform(fps_xstar_1)
        fps_xstar_pca_3 = np.hstack((fps_xstar_pca_3, x_output_3[:, None]))
        fps_xstar_pca_4 = pca_phase.transform(fps_xstar_4)
        fps_xstar_pca_4 = np.hstack((fps_xstar_pca_4, x_output_4[:, None]))

        # Initialize a 1x4 subplot
        N = len(phases)

        # Initialize a 1xN subplot
        fig = sp.make_subplots(
            rows=1,
            cols=N,
            subplot_titles=list(phases),
            specs=[[{"type": "scatter3d"}] * N],
        )

        # Create a DataFrame for Plotly
        df = []
        df.append(pd.DataFrame(fps_xstar_pca_1, columns=["PC1", "PC2", "Output"]))
        df.append(pd.DataFrame(fps_xstar_pca_2, columns=["PC1", "PC2", "Output"]))
        df.append(pd.DataFrame(fps_xstar_pca_3, columns=["PC1", "PC2", "Output"]))
        df.append(pd.DataFrame(fps_xstar_pca_4, columns=["PC1", "PC2", "Output"]))

        phase_names = ["context", "stim1", "mem1", "response"]

        for i in range(N):
            trace = px.scatter_3d(
                df[i],
                x="PC1",
                y="PC2",
                z="Output",
                opacity=0.3,
                title="Task trained fixed points",
            ).data[0]
            trace.marker.color = "black"  # Set marker color to black
            fig.add_trace(trace, row=1, col=i + 1)

        # Transform and plot each trajectory in each subplot
        for j in range(N):
            phase_name = phase_names[j]
            for i, trajectory in enumerate(phase_lats_lagged[phase_name]):
                trajectory_pca = pca_phase.transform(trajectory.detach().numpy())
                trial_output = phase_outs_lagged[phase_name][i][:, 1].detach().numpy()
                trajectory_outputs = np.hstack((trajectory_pca, trial_output[:, None]))
                df_trajectory = pd.DataFrame(
                    trajectory_outputs, columns=["PC1", "PC2", "Output"]
                )

                fig.add_trace(
                    px.line_3d(df_trajectory, x="PC1", y="PC2", z="Output").data[0],
                    row=1,
                    col=j + 1,
                )
        axis_limits = [-5, 5]
        # Update traces and layout for each subplot
        for i in range(N):
            fig.update_traces(marker=dict(size=5), row=1, col=i + 1)
            fig.update_scenes(
                xaxis_nticks=4,
                xaxis_range=axis_limits,
                yaxis_nticks=4,
                yaxis_range=axis_limits,
                zaxis_nticks=4,
                zaxis_range=[-1.1, 1.1],
                # aspectmode="cube",  # This makes the aspect ratio of the plot cubic
                row=1,
                col=i + 1,
            )

            # Add title for each subplot
            fig.add_annotation(
                text=phase_names[i],  # Subplot Title
                x=i / N + 0.5 / N,  # Position the title in the middle of each subplot
                y=1,  # Position the title above the subplot
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )

        # Update the overall layout if needed
        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=30)
        )  # Increase top margin to make space for titles

        # Update the overall layout if needed
        fig.update_layout(margin=dict(r=0, l=0, b=0, t=0))

        # Save as interactive HTML
        fig.show()
        fig.write_html(f"{self.plot_path}{self.suffix}_tt_fps.html")
        # Save as png
        fig.write_image(f"{self.plot_path}{self.suffix}_tt_fps.png")

    def interpolate_FPs_MultiTask(self):
        # Compute latent activity from task trained model

        self.tt_model.eval()

        data_dict = self.tt_datamodule.all_data
        # inputs = data_dict["inputs"]
        ics = data_dict["ics"]
        # targets = data_dict["targets"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        true_inputs = data_dict["true_inputs"]

        tt_latents = []
        tt_outputs = []

        task_to_analyze = "MemoryPro"
        phase_to_interpolate = "response"
        input_to_interpolate = "Fixation"
        num_interpolations = 10

        # find indices where task_to_analyze is the task
        task_inds = [i for i, task in enumerate(task_names) if task == task_to_analyze]

        task_inputs = torch.Tensor(true_inputs[task_inds])
        task_ics = torch.Tensor(ics[task_inds])
        task_phase_dict = [
            dict1 for i, dict1 in enumerate(phase_dict) if i in task_inds
        ]

        tt_outputs, tt_latents, _ = self.task_train_wrapper(task_ics, task_inputs)
        readout = self.task_train_wrapper.model.readout
        phases = task_phase_dict[0].keys()
        phase_for_pca = "mem1"

        phase_lats = {}
        phase_outs = {}
        phase_outs_lagged = {}
        phase_lats_lagged = {}
        phase_inputs = {}
        phase_lats_tensor = {}
        phase_outs_tensor = {}
        phase_inputs_tensor = {}

        for phase in phases:
            phase_lats[phase] = []
            phase_outs[phase] = []
            phase_inputs[phase] = []
            phase_lats_lagged[phase] = []
            phase_outs_lagged[phase] = []
            for i, phase_dict in enumerate(task_phase_dict):
                phase_start = phase_dict[phase][0]
                phase_end = phase_dict[phase][1]
                phase_lats[phase].append(tt_latents[i, phase_start:phase_end, :])
                phase_inputs[phase].append(task_inputs[i, phase_start:phase_end, :])
                phase_outs[phase].append(tt_outputs[i, phase_start:phase_end, :])
                if phase_start > 0:
                    phase_lats_lagged[phase].append(
                        tt_latents[i, phase_start - 2 : phase_end, :]
                    )
                    phase_outs_lagged[phase].append(
                        tt_outputs[i, phase_start - 2 : phase_end, :]
                    )
                else:
                    phase_lats_lagged[phase].append(
                        tt_latents[i, phase_start:phase_end, :]
                    )
                    phase_outs_lagged[phase].append(
                        tt_outputs[i, phase_start:phase_end, :]
                    )

            # Stack along first dimension and compute FP
            phase_lats_tensor[phase] = torch.vstack(phase_lats[phase])
            phase_inputs_tensor[phase] = torch.vstack(phase_inputs[phase])
            phase_outs_tensor[phase] = torch.vstack(phase_outs[phase])

        pca_phase = PCA(n_components=2)
        pca_phase.fit(phase_lats_tensor[phase_for_pca].detach().numpy())

        # Panel 1: FP find during Context and Memory phases
        panel1_lats = phase_lats_tensor[phase_to_interpolate]
        panel1_inputs = phase_inputs_tensor[phase_to_interpolate]

        interp_input_index = self.tt_datamodule.data_env.input_labels.index(
            input_to_interpolate
        )
        start_val = 1.0
        end_val = phase_inputs[phase_to_interpolate][0][-1, interp_input_index]
        interp_vals = np.linspace(start_val, end_val, num_interpolations)
        interp_fps = []
        inter_x_star = []
        readout_proj = []
        n_inits = 4096
        for i in range(num_interpolations):
            print(f"Interpolating {i} of {num_interpolations}")
            panel1_inputs[:, interp_input_index] = interp_vals[i]
            fps = find_fixed_points(
                model=self.task_train_wrapper,
                state_trajs=panel1_lats,
                mode="tt",
                n_inits=n_inits,
                noise_scale=0.5,
                learning_rate=1e-2,
                tol_q=1e-5,
                tol_dq=1e-20,
                tol_unique=1e-6,
                max_iters=10000,
                random_seed=0,
                do_fp_sort=False,
                device="cpu",
                seed=0,
                inputs=panel1_inputs,
                compute_jacobians=False,
            )
            interp_fps.append(fps)
            inter_x_star.append(fps.xstar)
            readout_proj.append(readout(torch.Tensor(fps.xstar)))

        # Initialize plot
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

        # Colors for the points
        colors = px.colors.qualitative.Plotly

        # Create scatter plot for each set of points
        for i in range(num_interpolations):
            inter_x_pca = pca_phase.transform(inter_x_star[i])
            scatter = go.Scatter3d(
                x=inter_x_pca[:, 0],
                y=inter_x_pca[:, 1],
                z=readout_proj[i][:, 1],
                mode="markers",
                marker=dict(
                    size=5, color=colors[i % len(colors)]
                ),  # Adjust size as needed
                name=f"Interpolation {i}",
            )
            fig.add_trace(scatter)

        # Update layout if necessary
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title="X Axis Title"),  # Update axis titles as necessary
                yaxis=dict(title="Y Axis Title"),
                zaxis=dict(title="Z Axis Title"),
            ),
        )

        # Show plot
        fig.show()

        # Save as HTML
        fig.write_html(f"{self.plot_path}{self.suffix}_tt_fps_interp.html")


class Comparator_TT:
    # TODO
    def __init__(self, suffix):
        self.task_train_wrapper_1 = None
        self.task_train_wrapper_2 = None
        self.env_1 = None
        self.env_2 = None
        self.tt_model_1 = None
        self.tt_model_2 = None
        self.tt_datamodule_1 = None
        self.tt_datamodule_2 = None
        self.suffix = suffix
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{suffix}/"
        )

        self.latents_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            "interpretability/comparison/latents/"
        )
        self.tt_fps_1 = None
        self.tt_fps_2 = None

    def load_task_train_wrapper_1(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.task_train_wrapper_1 = pickle.load(f)
        self.env_1 = self.task_train_wrapper_1.task_env
        self.tt_model_1 = self.task_train_wrapper_1.model
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.tt_datamodule_1 = pickle.load(f)
            self.tt_datamodule_1.prepare_data()
            self.tt_datamodule_1.setup()

    def load_task_train_wrapper_2(self, filepath):
        with open(filepath + "model.pkl", "rb") as f:
            self.task_train_wrapper_2 = pickle.load(f)
        self.env_2 = self.task_train_wrapper_2.task_env
        self.tt_model_2 = self.task_train_wrapper_2.model
        with open(filepath + "datamodule.pkl", "rb") as f:
            self.tt_datamodule_2 = pickle.load(f)
            self.tt_datamodule_2.prepare_data()
            self.tt_datamodule_2.setup()

    def perform_dsa(self):
        # Compute latent activity from task trained model

        self.tt_model_1.eval()
        tt_train_ds_1 = self.tt_datamodule_1.train_ds
        tt_val_ds_1 = self.tt_datamodule_1.valid_ds
        tt_ics_1 = torch.vstack((tt_train_ds_1.tensors[0], tt_val_ds_1.tensors[0]))
        tt_inputs_1 = torch.vstack((tt_train_ds_1.tensors[1], tt_val_ds_1.tensors[1]))
        tt_model_out_1, tt_latents_1, tt_actions_1 = self.task_train_wrapper_1(
            tt_ics_1, tt_inputs_1
        )
        tt_latents_1 = tt_latents_1.detach().numpy()

        self.tt_model_2.eval()
        tt_train_ds_2 = self.tt_datamodule_2.train_ds
        tt_val_ds_2 = self.tt_datamodule_2.valid_ds
        tt_ics_2 = torch.vstack((tt_train_ds_2.tensors[0], tt_val_ds_2.tensors[0]))
        tt_inputs_2 = torch.vstack((tt_train_ds_2.tensors[1], tt_val_ds_2.tensors[1]))
        tt_model_out_2, tt_latents_2, tt_actions_2 = self.task_train_wrapper_2(
            tt_ics_2, tt_inputs_2
        )
        tt_latents_2 = tt_latents_2.detach().numpy()

        dsa = DSA(
            X=tt_latents_1,
            Y=tt_latents_2,
            n_delays=10,
            rank=30,
            delay_interval=5,
            verbose=True,
            iters=1000,
            lr=1e-2,
        )
        similarities = dsa.fit_score()
        return similarities
