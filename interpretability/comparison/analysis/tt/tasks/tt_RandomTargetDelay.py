import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.tt.tt import Analysis_TT


class TT_RandomTargetDelay(Analysis_TT):
    def __init__(self, run_name, filepath):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath)
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{self.run_name}/"
        )

    def plotTrial(self, trial_num):
        # plot the trial
        # get the trial
        tt_ics, tt_inputs, tt_targets = self.get_model_input()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        controlled = out_dict["controlled"]
        states = out_dict["states"]

        targets_trial = tt_targets[trial_num, :, :].detach().numpy()
        controlled_trial = controlled[trial_num, :, :].detach().numpy()

        # plot the trial
        fig = plt.figure(figsize=(5, 8))
        # Put a square at the start location (targets_trial)
        ax = fig.add_subplot(311)
        ax.plot(
            targets_trial[0, 0],
            targets_trial[0, 1],
            marker="s",
            color="r",
            markersize=10,
        )
        ax.plot(
            targets_trial[-1, 0],
            targets_trial[-1, 1],
            marker="s",
            color="g",
            markersize=8,
        )

        ax.plot(
            controlled_trial[:, 0],
            controlled_trial[:, 1],
            color="k",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X Position (cm)")
        ax.set_ylabel("Y Position (cm)")
        ax.set_title(f"Trial {trial_num}")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        # Tight layout

        ax = fig.add_subplot(312)
        # Plot the x, y velocity over the trial
        ax.plot(np.diff(controlled_trial[:, 0]), label="x velocity")
        ax.plot(np.diff(controlled_trial[:, 1]), label="y velocity")
        ax.legend()
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Velocity (cm/bin)")

        # Plot the muscle activations over the trial
        ax = fig.add_subplot(313)
        ax.plot(out_dict["actions"][trial_num, :, :].detach().numpy())
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Muscle Activation (AU)")

        plt.tight_layout()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(211)

        ax.plot(states[trial_num, :, 2].detach().numpy(), "g", label="Vision")
        ax.plot(states[trial_num, :, 3].detach().numpy(), "g")
        for i in range(4, 10):
            if i == 4:
                ax.plot(
                    states[trial_num, :, i].detach().numpy(),
                    color="r",
                    label="Muscle Lengths",
                )
            else:
                ax.plot(states[trial_num, :, i].detach().numpy(), color="r")

        for i in range(10, 16):
            if i == 10:
                ax.plot(
                    states[trial_num, :, i].detach().numpy(),
                    color="b",
                    label="Muscle Velocity",
                )
            else:
                ax.plot(states[trial_num, :, i].detach().numpy(), color="b")
        ax.set_ylabel("State inputs")
        ax.set_xlim([0, 250])

        ax.legend(loc="upper right")

        ax = fig.add_subplot(212)

        ax.plot(tt_inputs[trial_num, :, 0].detach().numpy(), "k", label="Target onset")
        ax.plot(tt_inputs[trial_num, :, 1].detach().numpy(), "k")

        ax.plot(states[trial_num, :, 0].detach().numpy(), "g", label="go_cue")
        ax.plot(states[trial_num, :, 1].detach().numpy(), "g")

        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Goal Inputs")
        ax.set_xlim([0, 250])
        ax.legend(loc="lower right")

    def generate_latent_video(
        self,
        align_to="go_cue",
        color_by="target",
        pre_window=10,
        post_window=10,
        fps=10,
    ):
        tt_ics, tt_inputs, tt_targets = self.get_model_input()
        extra = self.get_extra_input()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        latents = out_dict["latents"]

        start_location = tt_targets[:, 0, :].detach().numpy()
        target_location = tt_targets[:, -1, :].detach().numpy()

        target_on = extra[:, 0]
        go_cue = extra[:, 1]

        # align latents to "align_to" variable
        if align_to == "go_cue":
            align_to_vec = go_cue
        elif align_to == "target_on":
            align_to_vec = target_on
        else:
            raise ValueError("align_to must be 'go_cue' or 'target_on'")

        n_trials, n_time, n_lats = latents.shape
        latents_aligned = np.zeros((n_trials, pre_window + post_window, n_lats))
        for i in range(n_trials):
            align_ind = align_to_vec[i].detach().numpy()
            align_ind = align_ind.astype(int)

            window_start = np.min(align_ind - pre_window, 0)
            window_end = np.max(align_ind + post_window, n_time)
            window_start = int(window_start)
            window_end = int(window_end)

            latents_aligned[i, :, :] = (
                latents[i, window_start:window_end, :].detach().numpy()
            )

        latents_aligned_flat = latents_aligned.reshape(-1, latents_aligned.shape[-1])
        pca = PCA(n_components=3)
        latents_aligned_pca = pca.fit_transform(latents_aligned_flat)
        latents_aligned_pca = latents_aligned_pca.reshape(
            latents_aligned.shape[0], latents_aligned.shape[1], 3
        )

        def map_to_color(positions):
            # Normalize each dimension [0, 1]
            norm_x = plt.Normalize(positions[:, 0].min(), positions[:, 0].max())
            norm_y = plt.Normalize(positions[:, 1].min(), positions[:, 1].max())

            # Apply two different colormaps
            colormap_x = cm.jet(norm_x(positions[:, 0]))
            colormap_y = cm.jet(norm_y(positions[:, 1]))

            # Combine colors (e.g., by averaging)
            combined_color = (colormap_x + colormap_y) / 2

            return combined_color

        if color_by == "target":
            colors = map_to_color(target_location)
        elif color_by == "start":
            colors = map_to_color(start_location)

        # Make a frame for each time point
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        # Initialize the video writer
        writer = FFMpegWriter(fps=fps)

        # Create a video file and write frames
        with writer.saving(fig, f"{color_by}_{align_to}_latent_video.mp4", 100):
            for t in range(pre_window + post_window):
                print(f"Writing frame {t} of {pre_window + post_window}")
                ax.clear()
                for i in range(n_trials):
                    ax.scatter(
                        latents_aligned_pca[i, t, 0],
                        latents_aligned_pca[i, t, 1],
                        latents_aligned_pca[i, t, 2],
                        color=colors[i],
                    )
                ax.set_title(f"Time {t-pre_window}")
                ax.set_xlim([-5, 5])  # Set this according to your data range
                ax.set_ylim([-5, 5])  # Set this according to your data range
                ax.set_zlim([-5, 5])
                writer.grab_frame()
