import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


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

    def compare_latents(self):
        # Function to compare the latent activity
        ij_list = []
        ji_list = []
        for i in range(self.num_analyses):
            for j in range(i + 1, self.num_analyses):
                i_j_r2, j_i_r2 = self.compare_latents_pair(
                    self.analyses[i], self.analyses[j]
                )
                ij_list.append(i_j_r2)
                ji_list.append(j_i_r2)
        return ij_list, ji_list

    def compare_latents_pair(self, analysis1, analysis2, num_pcs=3):
        lats1 = analysis1.get_latents()
        lats1_flat = (
            lats1.reshape(lats1.shape[0] * lats1.shape[1], lats1.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA()
        lats1_flat_pca = pca.fit_transform(lats1_flat)
        lats1 = lats1_flat_pca.reshape(lats1.shape)

        lats2 = analysis2.get_latents()
        lats2_flat = (
            lats2.reshape(lats2.shape[0] * lats2.shape[1], lats2.shape[2])
            .detach()
            .numpy()
        )
        pca = PCA()
        lats2_flat_pca = pca.fit_transform(lats2_flat)
        lats2 = lats2_flat_pca.reshape(lats2.shape)

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

    def compare_dynamics_DSA(self):
        # Function to compare the dynamics using DSA
        pass

    def plot_trials(self, num_trials, num_pcs=3):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            print(latents.shape)
            pca = PCA()
            latents_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            latents_pca_flat = pca.fit_transform(latents_flat)
            latents_pca = latents_pca_flat.reshape(latents.shape)
            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
