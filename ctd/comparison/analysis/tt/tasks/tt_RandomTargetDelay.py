import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from ctd.comparison.analysis.tt.tt import Analysis_TT


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

    def plot_trial(self, trial_num):
        # plot the trial
        # get the trial
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        inputs_to_env = self.get_inputs_to_env()
        tt_extra = self.get_extra_inputs()
        out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env=inputs_to_env)
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
            # dashed line
            linestyle="--",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X Position (cm)")
        ax.set_ylabel("Y Position (cm)")
        ax.set_title(f"Trial {trial_num}")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-0.2, 1.2])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 1])
        # Tight layout

        ax = fig.add_subplot(312)
        # Plot the x, y velocity over the trial
        ax.plot(np.diff(controlled_trial[:, 0]), label="x velocity")
        ax.plot(np.diff(controlled_trial[:, 1]), label="y velocity")
        ax.legend()
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Velocity (cm/bin)")
        ax.set_ylim([-0.05, 0.05])

        # Plot the muscle activations over the trial
        ax = fig.add_subplot(313)
        ax.plot(out_dict["actions"][trial_num, :, :].detach().numpy())
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Muscle Activation (AU)")
        ax.set_xlim([0, 300])
        ax.set_ylim([-0.0, 0.5])
        plt.tight_layout()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(311)
        ax.plot(states[trial_num, :, 0].detach().numpy(), "g", label="Vision")
        ax.plot(states[trial_num, :, 1].detach().numpy(), "g")

        for i in range(2, 8):
            if i == 2:
                ax.plot(
                    states[trial_num, :, i].detach().numpy(),
                    color="r",
                    label="Muscle Lengths",
                )
            else:
                ax.plot(states[trial_num, :, i].detach().numpy(), color="r")

        for i in range(8, 14):
            if i == 8:
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

        ax = fig.add_subplot(312)

        ax.plot(tt_inputs[trial_num, :, 0].detach().numpy(), "k", label="Target onset")
        ax.plot(tt_inputs[trial_num, :, 1].detach().numpy(), "k")

        ax.plot(states[trial_num, :, 0].detach().numpy(), "g", label="go_cue")
        ax.plot(states[trial_num, :, 1].detach().numpy(), "g")
        # Add a vertical line at the go cue
        target_on_ind = tt_extra[trial_num, 0].detach().numpy()
        go_cue_ind = tt_extra[trial_num, 1].detach().numpy()
        ax.axvline(target_on_ind, color="k", linestyle="--")
        ax.axvline(go_cue_ind, color="g", linestyle="--")

        ax.set_ylabel("Goal Inputs")
        ax.set_xlim([0, 300])
        ax.legend(loc="lower right")

        ax = fig.add_subplot(313)
        ax.plot(
            inputs_to_env[trial_num, :, 0].detach().numpy(), "k", label="Bump onset"
        )
        ax.plot(inputs_to_env[trial_num, :, 1].detach().numpy(), "k")
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Environment Inputs")

    def generate_latent_video(
        self,
        align_to="go_cue",
        dims_by="target",  # PCA, hand, target
        color_by="target",
        pre_window=10,
        post_window=10,
        fps=10,
    ):
        # Get model inputs
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        extra = self.get_extra_inputs()

        # Get model outputs
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        latents = out_dict["latents"]
        hand_pos = out_dict["controlled"]

        # Get the target and start locations
        start_location = tt_targets[:, 0, :].detach().numpy()
        target_location = tt_targets[:, -1, :].detach().numpy()

        # Get the go cue and target on times
        target_on = extra[:, 0]
        go_cue = extra[:, 1]

        go_trials = go_cue.detach().numpy() > 0

        # Make a hand position decoder
        lats_flat = latents.reshape(-1, latents.shape[-1]).detach().numpy()
        hand_pos_flat = hand_pos.reshape(-1, hand_pos.shape[-1]).detach().numpy()
        hand_pos_decoder = LinearRegression().fit(lats_flat, hand_pos_flat)

        # Make a target location decoder
        target_location_flat = (
            tt_targets.reshape(-1, tt_targets.shape[-1]).detach().numpy()
        )
        target_location_decoder = LinearRegression().fit(
            lats_flat, target_location_flat
        )

        # align latents to "align_to" variable
        if align_to == "go_cue":
            # Only use trials where the go cue has been given
            target_on = target_on[go_trials]
            start_location = start_location[go_trials]
            target_location = target_location[go_trials]
            latents = latents[go_trials]
            go_cue = go_cue[go_trials]
            hand_pos = hand_pos[go_trials]
            align_to_vec = go_cue

        elif align_to == "target_on":
            align_to_vec = target_on
        else:
            raise ValueError("align_to must be 'go_cue' or 'target_on'")

        # Align latents to the align_to variable
        n_trials, n_time, n_lats = latents.shape
        latents_aligned = []
        for i in range(n_trials):
            align_ind = align_to_vec[i].detach().numpy()
            align_ind = align_ind.astype(int)
            window_start = align_ind - pre_window
            window_end = align_ind + post_window
            window_start = int(window_start)
            window_end = int(window_end)
            if window_start > 0 and window_end < n_time:
                latents_aligned.append(
                    latents[i, window_start:window_end, :].detach().numpy()
                )
        latents_aligned = np.array(latents_aligned)

        latents_aligned_flat = latents_aligned.reshape(-1, latents_aligned.shape[-1])
        pca = PCA(n_components=3)
        latents_aligned_pca = pca.fit_transform(latents_aligned_flat)

        if dims_by == "PCA":
            latents_aligned_pca = latents_aligned_pca.reshape(
                latents_aligned.shape[0], latents_aligned.shape[1], 3
            )
        elif dims_by == "hand":
            latents_aligned_plot = hand_pos_decoder.predict(latents_aligned_flat)
            latents_aligned_plot = latents_aligned_plot.reshape(
                latents_aligned.shape[0], latents_aligned.shape[1], 2
            )
            first_pc = latents_aligned_pca[:, 0].reshape(
                latents_aligned.shape[0], latents_aligned.shape[1], 1
            )
            latents_aligned_plot = np.concatenate(
                (latents_aligned_plot, first_pc),
                axis=2,
            )

        elif dims_by == "target":
            latents_aligned_plot = target_location_decoder.predict(latents_aligned_flat)
            latents_aligned_plot = latents_aligned_plot.reshape(
                latents_aligned.shape[0], latents_aligned.shape[1], 2
            )
            first_pc = latents_aligned_pca[:, 0].reshape(
                latents_aligned.shape[0], latents_aligned.shape[1], 1
            )
            latents_aligned_plot = np.concatenate(
                (latents_aligned_plot, first_pc),
                axis=2,
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
        n_trials_plot = latents_aligned_plot.shape[0]
        # Create a video file and write frames
        with writer.saving(fig, f"{color_by}_{align_to}_latent_video.mp4", 100):
            for t in range(pre_window + post_window):
                print(f"Writing frame {t} of {pre_window + post_window}")
                ax.clear()
                for i in range(n_trials_plot):
                    ax.scatter(
                        latents_aligned_plot[i, t, 0],
                        latents_aligned_plot[i, t, 1],
                        latents_aligned_plot[i, t, 2],
                        color=colors[i],
                    )
                ax.set_title(f"Time {t-pre_window}")
                ax.set_xlim([-1, 1])  # Set this according to your data range
                ax.set_ylim([-1, 1])  # Set this according to your data range
                ax.set_zlim([-1, 1])
                writer.grab_frame()
