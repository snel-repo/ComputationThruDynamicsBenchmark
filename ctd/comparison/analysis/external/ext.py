import h5py
import matplotlib.pyplot as plt
import torch

from ctd.comparison.analysis.analysis import Analysis


class Analysis_Ext(Analysis):
    def __init__(self, run_name, filepath):
        self.run_name = run_name
        self.filepath = filepath

        self.train_true_rates = None
        self.train_true_latents = None
        self.eval_true_rates = None
        self.eval_true_latents = None

        self.load_data(filepath)

    def load_data(self, filepath):
        with h5py.File(filepath, "r") as h5file:
            # Check the fields
            print(h5file.keys())
            self.eval_rates = torch.Tensor(h5file["eval_rates"][()])
            self.eval_latents = torch.Tensor(h5file["eval_latents"][()])
            self.train_rates = torch.Tensor(h5file["train_rates"][()])
            self.train_latents = torch.Tensor(h5file["train_latents"][()])
            if "fixed_points" in h5file.keys():
                self.fixed_points = torch.Tensor(h5file["fixed_points"][()])
            else:
                self.fixed_points = None

    def get_latents(self, phase="all"):
        if phase == "train":
            return self.train_latents
        elif phase == "val":
            return self.eval_latents
        else:
            full_latents = torch.cat((self.train_latents, self.eval_latents), dim=0)
            return full_latents

    def get_rates(self, phase="all"):
        if phase == "train":
            return self.train_rates
        elif phase == "val":
            return self.eval_rates
        else:
            full_rates = torch.cat((self.train_rates, self.eval_rates), dim=0)
            return full_rates

    def get_true_rates(self, phase="all"):
        if phase == "train":
            return self.train_true_rates
        elif phase == "val":
            return self.eval_true_rates
        else:
            full_true_rates = torch.cat(
                (self.train_true_rates, self.eval_true_rates), dim=0
            )
            return full_true_rates

    def get_model_outputs(self, phase="all"):
        if phase == "train":
            return self.train_rates, self.train_latents
        elif phase == "val":
            return self.eval_rates, self.eval_latents
        else:
            return self.get_rates(), self.get_latents()

    def compute_FPs(self, latents, inputs):
        return None

    def add_true_rates(self, train_true_rates, eval_true_rates):
        self.train_true_rates = train_true_rates
        self.eval_true_rates = eval_true_rates

    def plot_fps(self):
        if self.fixed_points is None:
            print("No fixed points to plot")
            return
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        latents = self.get_latents(phase="val")
        fps = self.fixed_points
        for i in range(100):
            ax.plot(
                latents[i, :, 0],
                latents[i, :, 1],
                latents[i, :, 2],
                c="k",
                linewidth=0.1,
            )
        ax.scatter(fps[:, 0], fps[:, 1], fps[:, 2], c="r")
        ax.set_title(f"Fixed Points: {self.run_name}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
