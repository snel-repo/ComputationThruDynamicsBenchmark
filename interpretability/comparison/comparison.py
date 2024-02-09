import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from DSA import DSA
from interpretability.comparison.metrics import (
    get_latents_vaf,
    get_rate_r2,
    get_state_r2,
)


class Comparison:
    def __init__(self):
        self.num_analyses = 0
        self.analyses = []
        self.ref_ind = None

    def load_analysis(self, analysis, reference_analysis=False):
        self.analyses.append(analysis)
        self.num_analyses += 1
        if self.ref_ind is None and reference_analysis:
            self.ref_ind = self.num_analyses - 1
        elif self.ref_ind is not None and reference_analysis:
            # Throw an error
            raise ValueError("There is already a reference analysis")

    def compare_rate_r2(self):
        # Function to compare the rate-reconstruction of the different models
        state_r2_mat = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                state_r2_mat[i, j] = get_rate_r2(self.analyses[i], self.analyses[j])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot it as an image
        ax.imshow(state_r2_mat)
        ax.set_title("Rate R2 from i to j")
        # Set ticks
        ax.set_xticks(np.arange(self.num_analyses))
        ax.set_yticks(np.arange(self.num_analyses))
        ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    def compare_state_r2(self, num_pcs=3):
        # Function to compare the latent activity
        state_r2_mat = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                state_r2_mat[i, j] = get_state_r2(
                    self.analyses[i].get_latents(),
                    self.analyses[j].get_latents(),
                    num_pcs=num_pcs,
                )

        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(state_r2_mat)
        ij_ax.set_title("R2 from i to j")
        # Set ticks
        ij_ax.set_xticks(np.arange(self.num_analyses))
        ij_ax.set_yticks(np.arange(self.num_analyses))
        ij_ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ij_ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ij_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # Colorbar
        ij_figure.colorbar(
            plt.imshow(state_r2_mat),
            ax=ij_ax,
        )
        return state_r2_mat

    def compare_to_reference_affine(self, ref_ind=None):
        # Function to compare the latent activity
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        rate_state_mat = np.zeros((self.num_analyses, 2))
        for i in range(self.num_analyses):
            if i == ref_ind:
                continue
            rate_state_mat[i, 0] = get_state_r2(self.analyses[i], reference_analysis)
            rate_state_mat[i, 1] = get_state_r2(reference_analysis, self.analyses[i])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.num_analyses):
            if i != ref_ind:
                ax.scatter(rate_state_mat[i, 0], rate_state_mat[i, 1])
                ax.text(
                    rate_state_mat[i, 0],
                    rate_state_mat[i, 1],
                    str(i),
                )

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_title("Latent Similarity (Affine matching)")
        ax.set_xlabel("Model captures Reference (~ Rate R2)")
        ax.set_ylabel("Reference captures Model (~ State R2)")

    def compare_latents_vaf(self):
        # Function to compare the latent activity
        vaf = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                vaf[i, j] = get_latents_vaf(self.analyses[i], self.analyses[j])

        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(vaf)
        ij_ax.set_title("R2 from i to j")
        # Set ticks
        ij_ax.set_xticks(np.arange(self.num_analyses))
        ij_ax.set_yticks(np.arange(self.num_analyses))
        ij_ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ij_ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ij_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # Colorbar
        ij_figure.colorbar(plt.imshow(vaf), ax=ij_ax)
        return vaf

    def compare_dynamics_DSA(
        self, n_delays=20, rank=50, delay_interval=1, device="cpu"
    ):
        latent_list = []
        for analysis in self.analyses:
            latents = analysis.get_latents().detach().numpy()
            latent_list.append(latents.reshape(-1, latents.shape[-1]))

        dsa = DSA(
            latent_list,
            n_delays=n_delays,
            rank=rank,
            delay_interval=delay_interval,
            verbose=True,
            device=device,
            iters=1500,
            lr=0.005,
        )
        similarities = dsa.fit_score()

        fit_mat = similarities
        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(fit_mat)
        ij_ax.set_title("R2 from i to j")
        ij_ax.set_xticks(np.arange(self.num_analyses))
        ij_ax.set_yticks(np.arange(self.num_analyses))
        ij_ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ij_ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ij_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # Colorbar
        ij_figure.colorbar(plt.imshow(fit_mat), ax=ij_ax)
        return similarities

    def plot_trials(self, num_trials, num_pcs=3):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            pca = PCA()
            latents_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            latents_pca_flat = pca.fit_transform(latents_flat)
            latents_pca = latents_pca_flat.reshape(latents.shape)
            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_trials_reference(self, num_trials=2, ref_ind=None, num_pcs=3):

        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = self.analyses[ref_ind].get_latents().detach().numpy()
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure()
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)

            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")
