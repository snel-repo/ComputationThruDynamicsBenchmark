import matplotlib.pyplot as plt
import numpy as np
import torch
from DSA import DSA
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ctd.comparison.metrics import (
    get_latents_vaf,
    get_rate_r2,
    get_state_r2,
    get_state_r2_vaf,
)


class Comparison:
    def __init__(self, comparison_tag=None):
        self.comparison_tag = comparison_tag
        self.num_analyses = 0
        self.analyses = []
        self.ref_ind = None
        self.groups = []

    def load_analysis(self, analysis, group="None", reference_analysis=False):
        self.analyses.append(analysis)
        self.groups.append(group)
        self.num_analyses += 1
        if self.ref_ind is None and reference_analysis:
            self.ref_ind = self.num_analyses - 1
        elif self.ref_ind is not None and reference_analysis:
            # Throw an error
            raise ValueError("There is already a reference analysis")

    def regroup(self):
        groups = np.array(self.groups)
        # Sort analyses by group
        sorted_inds = np.argsort(groups)
        self.analyses = [self.analyses[i] for i in sorted_inds]
        self.groups = groups[sorted_inds]
        self.ref_ind = np.where(sorted_inds == self.ref_ind)[0][0]

    def compare_rate_r2(self):
        # Function to compare the rate-reconstruction of the different models
        dt_inds = []
        rate_r2_mat = np.zeros(self.num_analyses)
        for i in range(self.num_analyses):
            if self.analyses[i].tt_or_dt == "dt":
                dt_inds.append(i)
                rate_r2_mat[i] = get_rate_r2(
                    self.analyses[i].get_rates(), self.analyses[i].get_true_rates()
                )
        dt_inds = np.array(dt_inds).astype(int)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot it as an image
        ax.bar(np.arange(len(dt_inds)), rate_r2_mat[dt_inds])
        ax.set_title("Rate R2 for data-trained model")
        # Set ticks
        ax.set_xticks(np.arange(len(dt_inds)))
        ax.set_xticklabels(
            [
                analysis.run_name
                for analysis in self.analyses
                if analysis.tt_or_dt == "dt"
            ]
        )
        ax.set_ylabel("Rate R2")
        min_rate_r2 = np.min(rate_r2_mat[dt_inds])
        max_rate_r2 = np.max(rate_r2_mat[dt_inds])
        ax.set_ylim([min_rate_r2 - 0.05, max_rate_r2 + 0.05])
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    def compare_state_r2(self, num_pcs=4):
        # Function to compare the latent activity
        state_r2_mat = np.zeros((self.num_analyses, self.num_analyses))
        for i in range(self.num_analyses):

            for j in range(self.num_analyses):
                if i == j:
                    state_r2_mat[i, j] = 1
                else:
                    state_r2_mat[i, j] = get_state_r2(
                        self.analyses[i].get_latents(),
                        self.analyses[j].get_latents(),
                        num_pcs=num_pcs,
                    )

        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(state_r2_mat)
        ij_ax.set_title("State R2 from i to j")
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

    def compare_rate_state_r2(
        self,
        ref_ind=None,
        label_runs=False,
        label_groups=True,
        phase="val",
        save_pdf=False,
    ):
        # Function to compare the latent activity
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        rate_state_mat = np.zeros((self.num_analyses, 2))
        true_lats = reference_analysis.get_latents(phase=phase)
        is_multitask = reference_analysis.env == "MultiTask"
        if is_multitask:
            trial_lens = self.analyses[1].get_trial_lens(phase=phase)
            # Stack the latents to the different trial lengths
            true_lats_stack = []
            for i in range(trial_lens.shape[0]):
                true_lats_stack.append(true_lats[i, : int(trial_lens[i]), :])
            true_lats = torch.vstack(true_lats_stack)

        for i in range(self.num_analyses):
            print(
                f"Working on {i+1} of {self.num_analyses}: {self.analyses[i].run_name}"
            )
            print(f"Group: {self.groups[i]}")
            if i == ref_ind:
                continue
            rates, latents = self.analyses[i].get_model_outputs(phase=phase)
            true_rates = self.analyses[i].get_true_rates(phase=phase)

            rates_stack = []
            latents_stack = []
            true_rates_stack = []

            # If multitask (with different trial lengths)
            if is_multitask:
                trial_lens = self.analyses[i].get_trial_lens(phase=phase)
                # Stack the latents to the different trial lengths
                for j in range(latents.shape[0]):
                    latents_stack.append(latents[j, : int(trial_lens[j]), :])
                    rates_stack.append(rates[j, : int(trial_lens[j]), :])
                    true_rates_stack.append(true_rates[j, : int(trial_lens[j]), :])
                rates = torch.vstack(rates_stack)
                latents = torch.vstack(latents_stack)
                true_rates = torch.vstack(true_rates_stack)

            # Check that latents arenot NaN
            if np.isnan(latents.detach().numpy()).any():
                continue

            # Compute the rate R2 and state R2
            rate_state_mat[i, 0] = get_rate_r2(
                rates_true=true_rates,
                rates_pred=rates,
            )
            rate_state_mat[i, 1] = get_state_r2_vaf(
                lats_true=true_lats,
                lats_pred=latents,
            )
            print(f"Rate R2: {rate_state_mat[i, 0]}")
            print(f"State R2: {rate_state_mat[i, 1]}")

        # Sort results by the groups
        num_groups = len(np.unique(self.groups))
        colors = plt.cm.get_cmap("tab10", num_groups)
        color_inds = np.array(
            [np.where(np.unique(self.groups) == group)[0][0] for group in self.groups]
        )

        # Plot the results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.num_analyses):
            if i != ref_ind:
                ax.scatter(
                    rate_state_mat[i, 0],
                    rate_state_mat[i, 1],
                    color=colors(color_inds[i]),
                )
                if label_runs:
                    ax.text(
                        rate_state_mat[i, 0],
                        rate_state_mat[i, 1],
                        self.analyses[i].run_name,
                    )
        if label_groups:
            for i in range(num_groups):
                ax.scatter(
                    [],
                    [],
                    color=colors(i),
                    label=np.unique(self.groups)[i],
                )
            ax.legend()
        bool_idx = np.arange(self.num_analyses) != ref_ind
        min_val = np.min(rate_state_mat[bool_idx]) - 0.2
        max_val = np.max(rate_state_mat[bool_idx]) + 0.2
        max_val = np.min([max_val, 1.05])
        min_val = np.max([min_val, -1.05])
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.plot([min_val, max_val], [min_val, max_val], "k--")
        # Square axis
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Rate R2 ")
        ax.set_ylabel("State R2 ")
        ax.set_title(f"Rate-State R2 ({self.comparison_tag})")
        if save_pdf:
            plt.savefig(f"{self.comparison_tag}_rate_state_r2.pdf")
        return rate_state_mat

    def compare_to_reference_affine(self, ref_ind=None, num_pcs=4):
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
            rate_state_mat[i, 0] = get_state_r2(
                reference_analysis.get_latents(phase="val"),
                self.analyses[i].get_latents(phase="val"),
                num_pcs=num_pcs,
            )
            rate_state_mat[i, 1] = get_state_r2(
                reference_analysis.get_latents(),
                self.analyses[i].get_latents(),
                num_pcs=num_pcs,
            )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.num_analyses):
            if i != ref_ind:
                ax.scatter(rate_state_mat[i, 0], rate_state_mat[i, 1])
                ax.text(
                    rate_state_mat[i, 0],
                    rate_state_mat[i, 1],
                    self.analyses[i].run_name,
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
        self,
        n_delays=20,
        rank=50,
        delay_interval=1,
        device="cuda",
        percent_data=0.01,
    ):
        latent_list = []
        for analysis in self.analyses:
            latents = analysis.get_latents().detach().numpy()
            # latent_list.append(latents.reshape(-1, latents.shape[-1]))
            latent_list.append(latents[: int(percent_data * latents.shape[0]), :, :])

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
        cmap_r = plt.cm.get_cmap("viridis_r")
        fit_mat = similarities
        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(fit_mat, cmap=cmap_r)
        ij_ax.set_title("Dynamical Similarity (DSA)")
        ij_ax.set_xticks(np.arange(self.num_analyses))
        ij_ax.set_yticks(np.arange(self.num_analyses))
        ij_ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ij_ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ij_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # INvert the colorbar

        ij_figure.colorbar(
            plt.imshow(fit_mat, cmap=cmap_r),
            ax=ij_ax,
        )
        # Save as pdf in a different
        plt.savefig(f"{self.comparison_tag}_dsa.pdf")
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

    def plot_inputs(self, num_trials, num_pcs=3):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            inputs = self.analyses[i].get_inputs().detach().numpy()
            for j in range(num_trials):
                axes[i, j].plot(inputs[j, :, :])
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

    def plot_trials_3d(self, num_trials):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(1, self.num_analyses, subplot_kw={"projection": "3d"})
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            pca = PCA()
            latents_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            latents_pca_flat = pca.fit_transform(latents_flat)
            latents_pca = latents_pca_flat.reshape(latents.shape)
            for j in range(num_trials):
                axes[i].plot(
                    latents_pca[j, :, 0],
                    latents_pca[j, :, 1],
                    latents_pca[j, :, 2],
                )

            axes[i].set_title(f"{self.analyses[i].run_name}")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_zlabel("PC3")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])

    def plot_trials_3d_reference(self, num_trials, ref_ind=None):
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
        axes = fig.subplots(1, self.num_analyses, subplot_kw={"projection": "3d"})
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)
            for j in range(num_trials):
                axes[i].plot(
                    latents_pca[j, :, 0],
                    latents_pca[j, :, 1],
                    latents_pca[j, :, 2],
                )

            axes[i].set_title(f"{self.analyses[i].run_name}")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_zlabel("PC3")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])

    def compare_procrustes(self, ref_ind=None):
        # Function to compare the latent activity
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        pro_mat = np.zeros(self.num_analyses)
        ref_lats = reference_analysis.get_latents(phase="val").detach().numpy()
        for i in range(self.num_analyses):
            print(f"Working on {i+1} of {self.num_analyses}")
            if i == ref_ind:
                continue
            _, latents = self.analyses[i].get_model_outputs(phase="val")

            if ref_lats.shape[-1] != latents.shape[-1]:
                continue
            latents = latents.detach().numpy()
            pro_mat[i] = procrustes(
                ref_lats.reshape(-1, ref_lats.shape[-1]),
                latents.reshape(-1, latents.shape[-1]),
            )[2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for group in np.unique(self.groups):
            group_inds = np.where(self.groups == group)[0]
            ax.bar(group_inds, pro_mat[group_inds], label=group)
        # Square axis
        ax.set_ylabel("M2 (Disparity)")
        ax.set_title("Procrustes M2")
        ax.legend()
        plt.savefig(f"{self.comparison_tag}_procrustes.pdf")

    def compare_CCA(self, ref_ind=None, num_components=10, iters=500):
        # Function to compare the latent activity
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        cca_mat = np.zeros((self.num_analyses, num_components))
        ref_lats = reference_analysis.get_latents(phase="val").detach().numpy()
        ref_lats = ref_lats.reshape(-1, ref_lats.shape[-1])
        for i in range(self.num_analyses):
            print(f"Working on {i+1} of {self.num_analyses}")
            if i == ref_ind:
                continue
            _, latents = self.analyses[i].get_model_outputs(phase="val")

            if ref_lats.shape[-1] != latents.shape[-1]:
                continue
            latents = latents.detach().numpy().reshape(-1, latents.shape[-1])
            cca = CCA(n_components=num_components, max_iter=iters)
            cca.fit(latents, ref_lats)
            cca_lats, cca_ref_lats = cca.transform(latents, ref_lats)
            cca_mat[i, :] = r2_score(cca_lats, cca_ref_lats, multioutput="raw_values")

        group_inds = np.unique(self.groups)
        colors = plt.cm.get_cmap("tab10", len(group_inds))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.num_analyses):
            ax.plot(
                np.arange(num_components),
                cca_mat[i, :],
                label=self.analyses[i].run_name,
                color=colors(np.where(group_inds == self.groups[i])[0][0]),
            )
        # Square axis
        plt.savefig(f"{self.comparison_tag}_CCA.pdf")

    def compare_performance(self):
        _, _, targets = self.analyses[self.ref_ind].get_model_inputs(phase="val")
        mean_r2 = []
        for i in range(self.num_analyses):
            print(f"Working on {i+1} of {self.num_analyses}")
            latents = self.analyses[i].get_latents(phase="val").detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            targets_flat = targets.reshape(
                targets.shape[0] * targets.shape[1], targets.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, targets_flat)
            pred = reg.predict(lats_flat)
            r2_perf = r2_score(targets_flat, pred, multioutput="raw_values")
            print(f"Performance R2s for {self.analyses[i].run_name} is {r2_perf}")
            mean_r2.append(np.mean(r2_perf))

        # Find the mean in each group
        mean_in_group = []
        for group in np.unique(self.groups):
            group_inds = np.where(self.groups == group)[0]
            mean_in_group.append(np.mean([mean_r2[i] for i in group_inds]))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(np.unique(self.groups))), mean_in_group)
        ax.set_xticks(np.arange(len(np.unique(self.groups))))
        ax.set_xticklabels(np.unique(self.groups))
        minVal = np.min(mean_in_group) - 0.05
        maxVal = np.max(mean_in_group) + 0.05
        ax.set_ylim([minVal, maxVal])
        ax.set_ylabel("Mean R2 of Performance in group")
        ax.set_title("Performance of data-trained models at task (R2)")
