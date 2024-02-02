import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from DSA import DSA


class Comparison:
    def __init__(self):
        self.num_analyses = 0
        self.analyses = []

    def load_analysis(self, analysis):
        self.analyses.append(analysis)
        self.num_analyses += 1

    def compare_recon(self):
        # Function to compare the rate-reconstruction of the different models
        pass

    def compare_latents_vaf(self):
        # Function to compare the latent activity
        vaf = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                vaf[i, j] = self.compare_latents_pair_vaf(
                    self.analyses[i], self.analyses[j]
                )

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

    def compare_latents_pair_vaf(self, analysis1, analysis2, num_pcs=3):
        lats1 = analysis1.get_latents()
        lats1_flat = (
            lats1.reshape(lats1.shape[0] * lats1.shape[1], lats1.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA(n_components=num_pcs)
        lats1_flat_pca = pca.fit_transform(lats1_flat)

        lats2 = analysis2.get_latents()
        lats2_flat = (
            lats2.reshape(lats2.shape[0] * lats2.shape[1], lats2.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA(n_components=num_pcs)
        lats2_flat_pca = pca.fit_transform(lats2_flat)
        reg = LinearRegression().fit(lats1_flat_pca, lats2_flat_pca)
        preds = reg.predict(lats1_flat_pca)
        var_exp = explained_variance_score(
            lats2_flat_pca, preds, multioutput="variance_weighted"
        )
        return var_exp

    def compare_latents(self):
        # Function to compare the latent activity
        ij_list = []
        ji_list = []

        fit_mat = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                i_j_r2, j_i_r2 = self.compare_latents_pair(
                    self.analyses[i], self.analyses[j]
                )
                ij_list.append(i_j_r2)
                ji_list.append(j_i_r2)
                fit_mat[i, j] = i_j_r2.mean()
        ij_list = np.array(ij_list)
        ji_list = np.array(ji_list)

        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(fit_mat)
        ij_ax.set_title("R2 from i to j")
        return ij_list, ji_list

    def compare_latents_pair(self, analysis1, analysis2, num_pcs=3):
        lats1 = analysis1.get_latents()
        lats1_flat = (
            lats1.reshape(lats1.shape[0] * lats1.shape[1], lats1.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA(n_components=num_pcs)
        lats1_flat_pca = pca.fit_transform(lats1_flat)
        lats1 = lats1_flat_pca.reshape((lats1.shape[0], lats1.shape[1], num_pcs))

        lats2 = analysis2.get_latents()
        lats2_flat = (
            lats2.reshape(lats2.shape[0] * lats2.shape[1], lats2.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA(n_components=num_pcs)
        lats2_flat_pca = pca.fit_transform(lats2_flat)
        lats2 = lats2_flat_pca.reshape((lats2.shape[0], lats2.shape[1], num_pcs))

        # Compare the latent activity
        i_j_r2 = []
        j_i_r2 = []
        for j in range(lats2.shape[-1]):
            reg = LinearRegression().fit(lats1_flat_pca, lats2_flat_pca[:, j])
            i_j_r2.append(reg.score(lats1_flat_pca, lats2_flat_pca[:, j]))
        for i in range(lats1.shape[-1]):
            reg = LinearRegression().fit(lats2_flat_pca, lats1_flat_pca[:, i])
            j_i_r2.append(reg.score(lats2_flat_pca, lats1_flat_pca[:, i]))

        i_j_r2 = np.array(i_j_r2)
        j_i_r2 = np.array(j_i_r2)

        i_j_r2 = i_j_r2[:num_pcs]
        j_i_r2 = j_i_r2[:num_pcs]

        print(f"Comparison for {analysis1.run_name} and {analysis2.run_name}")
        print(f"{analysis1.run_name} to {analysis2.run_name}: {i_j_r2.mean()}")
        print(f"{analysis2.run_name} to {analysis1.run_name}: {j_i_r2.mean()}")
        return i_j_r2, j_i_r2

    def compare_fixedpoints(self):
        # Function to compare the fixed points
        pass

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

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

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
