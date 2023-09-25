import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# import linear regression from SKLearn
from sklearn.linear_model import LinearRegression

from interpretability.comparison.fixedpoints import find_fixed_points

#


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
            "interpretability/comparison/plots/"
        )

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

    def plotLatentActivity(self):
        # Compute latent activity from task trained model
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
        tt_outputs = torch.vstack((tt_train_ds.tensors[2], tt_val_ds.tensors[2]))
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

            ax = fig.add_subplot(3, n_trials, i + 1 + n_trials)
            ax.plot(tt_latents_val[i, :, 0])
            ax.plot(tt_latents_val[i, :, 1])
            ax.plot(tt_latents_val[i, :, 2])

            ax = fig.add_subplot(3, n_trials, i + 1 + 2 * n_trials)
            ax.plot(dt_latents[i, :, 0])
            ax.plot(dt_latents[i, :, 1])
            ax.plot(dt_latents[i, :, 2])
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

            ax = fig.add_subplot(3, n_trials, i + 1 + n_trials)
            ax.plot(tt_latents_val[i, :, 0])
            ax.plot(tt_latents_val[i, :, 1])
            ax.plot(tt_latents_val[i, :, 2])

            ax = fig.add_subplot(3, n_trials, i + 1 + 2 * n_trials)
            ax.plot(dt_latents[i, :, 0])
            ax.plot(dt_latents[i, :, 1])
            ax.plot(dt_latents[i, :, 2])
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
            dt_latents = pickle.load(f)
        with open(ttLatentsPath, "rb") as f:
            tt_latents = pickle.load(f)

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

        # Fit linear regression between the latents
        lr_getSys = LinearRegression().fit(dt_latents_flat, tt_latents_flat)
        r2_getSys = lr_getSys.score(dt_latents_flat, tt_latents_flat)
        lr_noExtra = LinearRegression().fit(tt_latents_flat, dt_latents_flat)
        r2_noExtra = lr_noExtra.score(tt_latents_flat, dt_latents_flat)

        print(f"R2 for getting system: {r2_getSys}")
        print(f"R2 for no extra: {r2_noExtra}")

        lr_getSysPCA = LinearRegression().fit(dt_lats_pca, tt_lats_pca)
        r2_getSysPCA = lr_getSysPCA.score(dt_lats_pca, tt_lats_pca)
        lr_noExtraPCA = LinearRegression().fit(tt_lats_pca, dt_lats_pca)
        r2_noExtraPCA = lr_noExtraPCA.score(tt_lats_pca, dt_lats_pca)

        print(f"R2 for getting system: {r2_getSysPCA}")
        print(f"R2 for no extra: {r2_noExtraPCA}")

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
        return return_dict

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
            pickle.dump(tt_latents_val, f)
        with open(self.latents_path + self.suffix + "_dt_latents.pkl", "wb") as f:
            pickle.dump(dt_latents, f)

    def compareLatentActivity(self):
        # Compute latent activity from task trained model
        self.tt_model.eval()
        tt_train_ds = self.tt_datamodule.train_ds
        tt_val_ds = self.tt_datamodule.valid_ds
        tt_inputs = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
        tt_model_out, tt_latents = self.task_train_wrapper(tt_inputs)
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
        print(f"R2 for getting system: {r2_getSys}")
        print(f"R2 for no extra: {r2_noExtra}")

        lr_getSysPCA = LinearRegression().fit(dt_lats_pca, tt_lats_pca)
        r2_getSysPCA = lr_getSysPCA.score(dt_lats_pca, tt_lats_pca)
        lr_noExtraPCA = LinearRegression().fit(tt_lats_pca, dt_lats_pca)
        r2_noExtraPCA = lr_noExtraPCA.score(tt_lats_pca, dt_lats_pca)
        print(f"{self.suffix} Top 3 PCs only")
        print(f"R2 for getting system: {r2_getSysPCA}")
        print(f"R2 for no extra: {r2_noExtraPCA}")
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
                learning_rate=1e-2,
                tol_q=1e-6,
                tol_dq=1e-20,
                tol_unique=1e-2,
                max_iters=10000,
                random_seed=0,
                do_fp_sort=False,
                device="cpu",
                seed=0,
                input_cond=input_cond,
            )

            dt_fps = find_fixed_points(
                model=self.data_train_wrapper,
                state_trajs=train_dt_latents,
                mode="dt",
                n_inits=1024,
                noise_scale=0.01,
                learning_rate=1e-2,
                tol_q=1e-7,
                tol_dq=1e-20,
                tol_unique=1e-3,
                max_iters=10000,
                random_seed=0,
                do_fp_sort=False,
                device="cpu",
                seed=0,
                input_cond=input_cond,
            )

        self.tt_fps = tt_fps
        self.dt_fps = dt_fps
