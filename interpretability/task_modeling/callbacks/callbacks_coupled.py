import io
import os

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from sklearn.decomposition import PCA


def calculate_arm_positions(
    shoulder_angle, elbow_angle, upper_arm_length=1, forearm_length=1
):

    # Calculate elbow position relative to shoulder
    elbow_x = np.cos(shoulder_angle) * upper_arm_length
    elbow_y = np.sin(shoulder_angle) * upper_arm_length

    # Calculate hand position relative to elbow, considering the elbow angle
    hand_x = elbow_x + np.cos(shoulder_angle + elbow_angle) * forearm_length
    hand_y = elbow_y + np.sin(shoulder_angle + elbow_angle) * forearm_length

    return np.array([0, 0]), np.array([elbow_x, elbow_y]), np.array([hand_x, hand_y])


def draw_muscle_activation(ax, center, radius, start_angle, end_angle, color):
    """
    Draws an arc to represent muscle activation around a joint.
    """
    # Note: In matplotlib, 0 degrees is at the X-axis and angles increase clockwise.
    # We adjust angles because our definitions likely assume standard math angles
    # (counterclockwise from the positive X-axis).
    arc = patches.Arc(
        center,
        radius * 2,
        radius * 2,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=3,
    )
    ax.add_patch(arc)


def draw_biarticular_muscle(ax, start_pos, end_pos, isFlexor, color):
    """
    Draws a line to represent biarticular muscle activation, correctly shifted
    to indicate whether it is a flexor or extensor.
    """
    # Calculate the direction vector from start to end
    direction = end_pos - start_pos
    # Calculate the perpendicular vector
    perpendicular = np.array([-direction[1], direction[0]])

    if isFlexor:
        # Rotate the perpendicular vector CCW for flexors to shift the line to the left
        shift_vector = perpendicular * 0.1  # Scale the shift magnitude
    else:
        # Rotate the perpendicular vector CW for extensors
        # to shift the line to the right
        shift_vector = -perpendicular * 0.1  # Scale the shift magnitude

    # Apply the shift
    shifted_start_pos = start_pos + shift_vector
    shifted_end_pos = end_pos + shift_vector

    ax.plot(
        [shifted_start_pos[0], shifted_end_pos[0]],
        [shifted_start_pos[1], shifted_end_pos[1]],
        color=color,
        linewidth=2,
    )


# import PCA from sklearn.decomposition


def get_activation_color(activation_level):
    """
    Returns a color based on the activation level using a colormap.
    """
    cmap = plt.get_cmap("rainbow")
    return cmap(activation_level)


def fig_to_rgb_array(fig):
    # Convert the figure to a numpy array for TB logging
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        fig_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = fig_data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def get_wandb_logger(loggers):

    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            return logger.experiment
    else:
        return None


class LatentTrajectoryPlot(pl.Callback):
    def __init__(
        self,
        log_every_n_epochs=10,
    ):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = get_wandb_logger(trainer.loggers)

        # Get trajectories and model predictions
        train_dataloader = trainer.datamodule.train_dataloader()
        ics_train = torch.cat([batch[0] for batch in train_dataloader]).to(
            pl_module.device
        )
        inputs_train = torch.cat([batch[1] for batch in train_dataloader]).to(
            pl_module.device
        )
        targets_train = torch.cat([batch[2] for batch in train_dataloader]).to(
            pl_module.device
        )

        output_dict = pl_module.forward(ics_train, inputs_train, targets_train)

        lats_train = output_dict["latents"]

        lats_train = lats_train.detach().cpu().numpy()
        n_trials, n_times, n_lat_dim = lats_train.shape
        if n_lat_dim > 3:
            pca1 = PCA(n_components=3)
            lats_train = pca1.fit_transform(lats_train.reshape(-1, n_lat_dim))
            lats_train = lats_train.reshape(n_trials, n_times, 3)
            exp_var = np.sum(pca1.explained_variance_ratio_)

        # Plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in lats_train:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*lats_train[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*lats_train[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {exp_var:.2f}")
        plt.tight_layout()
        trainer.loggers[0].experiment.add_figure(
            "latent_trajectory", fig, global_step=trainer.global_step
        )
        logger.log(
            {"latent_traj": wandb.Image(fig), "global_step": trainer.global_step}
        )


class MotorNetVideoGeneration(pl.Callback):
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=10):

        self.log_every_n_epochs = log_every_n_epochs
        self.num_trials_to_plot = num_trials_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # Get trajectories and model predictions
        train_dataloader = trainer.datamodule.train_dataloader()
        # Get trajectories and model predictions
        batch = next(iter(train_dataloader))
        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)
        targets = batch[2].to(pl_module.device)

        output_dict = pl_module.forward(ics, inputs, targets)

        controlled = output_dict["controlled"]

        B, T, N = controlled.shape

        # Create a list to store frames
        frames = []

        # Prepare to log a video for all batches
        for i in range(self.num_trials_to_plot):
            for t in range(T):
                # Create a figure
                fig, ax = plt.subplots(figsize=(3, 3))

                input_temp = inputs[i, t, :].detach().cpu().numpy()
                # if input_temp is all zeros
                if input_temp.sum() != 0:
                    # Plot as a green square
                    input_patch = patches.Rectangle(
                        (inputs[i, t, 0].item() - 0.05, inputs[i, t, 1].item() - 0.05),
                        0.1,
                        0.1,
                        color="red",
                    )
                    ax.add_patch(input_patch)

                    # Plot target position as a red square
                    target = patches.Rectangle(
                        (
                            targets[i, t, 0].item() - 0.05,
                            targets[i, t, 1].item() - 0.05,
                        ),
                        0.1,
                        0.1,
                        color="green",
                    )
                    ax.add_patch(target)

                # Plot hand position as a yellow dot
                ax.scatter(
                    controlled[i, t, 0].item(),
                    controlled[i, t, 1].item(),
                    color="black",
                )

                # Draw a dashed line from the start position to the current position
                ax.plot(
                    controlled[i, :t, 0].detach().cpu().numpy(),
                    controlled[i, :t, 1].detach().cpu().numpy(),
                    color="black",
                    linestyle="--",
                )

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-0.2, 1.2)

                ax.set_xticks([])
                ax.set_yticks([])
                # Preserved distance aspect ratio
                ax.set_aspect("equal")
                ax.set_title(f"Trial {i} - Step {t} of {T}")
                # Save figure to a numpy array
                fig.canvas.draw()
                img_arr = np.array(fig.canvas.renderer.buffer_rgba())
                frames.append(img_arr)

                # Close figure to save memory
                plt.close(fig)

        # Save frames to a video using imageio
        video_path = "combined_video.mp4"
        imageio.mimwrite(video_path, frames, fps=40)

        # Log the video to wandb
        wandb.log({"video": wandb.Video(video_path)})

        # Remove the video file
        os.remove(video_path)


class MotorNetVideoGenerationArm(pl.Callback):
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=10):
        self.log_every_n_epochs = log_every_n_epochs
        self.num_trials_to_plot = num_trials_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # Get trajectories, model predictions, and actions
        train_dataloader = trainer.datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)
        targets = batch[2].to(pl_module.device)
        inputs_to_env = batch[6].to(pl_module.device)

        output_dict = pl_module.forward(
            ics[: self.num_trials_to_plot, :],
            inputs[: self.num_trials_to_plot, :, :],
            targets[: self.num_trials_to_plot, :, :],
            inputs_to_env[: self.num_trials_to_plot, :, :],
        )
        upper_arm_length = trainer.model.task_env.effector.skeleton.l1
        forearm_length = trainer.model.task_env.effector.skeleton.l2
        controlled = output_dict["controlled"]
        joints = output_dict["joints"]
        actions = output_dict[
            "actions"
        ]  # Assuming actions are included in the output_dict

        B, T, _ = controlled.shape

        # Create a list to store frames
        frames = []

        for i in range(self.num_trials_to_plot):
            for t in range(T):
                fig, ax = plt.subplots(figsize=(5, 5))

                shoulder_angle = joints[i, t, 0].item()  # Assuming angle is in degrees
                elbow_angle = joints[i, t, 1].item()  # Assuming angle is in degrees

                shoulder_angle_deg = np.rad2deg(shoulder_angle)
                elbow_angle_deg = np.rad2deg(elbow_angle)

                # Calculate arm segment positions
                shoulder_pos, elbow_pos, hand_pos = calculate_arm_positions(
                    shoulder_angle, elbow_angle, upper_arm_length, forearm_length
                )

                # Draw the arm
                ax.plot(
                    [shoulder_pos[0], elbow_pos[0]],
                    [shoulder_pos[1], elbow_pos[1]],
                    "k-",
                )  # Upper arm
                ax.plot(
                    [elbow_pos[0], hand_pos[0]], [elbow_pos[1], hand_pos[1]], "k-"
                )  # Forearm
                sho_mono_radius = 0.1
                elb_mono_radius = 0.05
                # Transform to degrees

                for muscle_idx in range(4):
                    activation_level = actions[i, t, muscle_idx].item()
                    color = get_activation_color(activation_level)
                    if muscle_idx == 0:
                        draw_muscle_activation(
                            ax,
                            shoulder_pos,
                            sho_mono_radius,
                            shoulder_angle_deg,
                            180,
                            color,
                        )
                    elif muscle_idx == 1:
                        draw_muscle_activation(
                            ax,
                            shoulder_pos,
                            sho_mono_radius,
                            0,
                            shoulder_angle_deg,
                            color,
                        )
                    elif muscle_idx == 2:
                        draw_muscle_activation(
                            ax,
                            elbow_pos,
                            elb_mono_radius,
                            shoulder_angle_deg + elbow_angle_deg,
                            shoulder_angle_deg + 180,
                            color,
                        )
                    elif muscle_idx == 3:
                        draw_muscle_activation(
                            ax,
                            elbow_pos,
                            elb_mono_radius,
                            shoulder_angle_deg + 180,
                            shoulder_angle_deg + elbow_angle_deg,
                            color,
                        )

                # Biarticular muscle visualization
                for muscle_idx in range(4, 6):
                    activation_level = actions[i, t, muscle_idx].item()
                    color = get_activation_color(activation_level)
                    if muscle_idx == 4:
                        draw_biarticular_muscle(
                            ax, shoulder_pos, elbow_pos, isFlexor=True, color=color
                        )
                    elif muscle_idx == 5:
                        draw_biarticular_muscle(
                            ax, shoulder_pos, elbow_pos, isFlexor=False, color=color
                        )
                input_temp = inputs[i, t, :].detach().cpu().numpy()

                # Draw a solid line a y = 0 to represent the ground
                ax.plot([-0.5, 0.5], [0, 0], "k-", linewidth=3)
                # if input_temp is all zeros
                if input_temp.sum() != 0:
                    # Plot as a green square
                    input_patch = patches.Rectangle(
                        (inputs[i, t, 0].item() - 0.05, inputs[i, t, 1].item() - 0.05),
                        0.1,
                        0.1,
                        color="red",
                    )
                    ax.add_patch(input_patch)

                    # Plot target position as a red square
                    target = patches.Rectangle(
                        (
                            targets[i, t, 0].item() - 0.05,
                            targets[i, t, 1].item() - 0.05,
                        ),
                        0.1,
                        0.1,
                        color="green",
                    )
                    ax.add_patch(target)

                # Plot hand position as a yellow dot
                ax.scatter(
                    controlled[i, t, 0].item(),
                    controlled[i, t, 1].item(),
                    color="black",
                )

                # Draw a dashed line from the start position to the current position
                ax.plot(
                    controlled[i, :t, 0].detach().cpu().numpy(),
                    controlled[i, :t, 1].detach().cpu().numpy(),
                    color="black",
                    linestyle="--",
                )

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-0.2, 1.2)
                ax.axis("off")  # Hide axes for better visualization

                ax.set_title(f"Trial {i}")
                # Draw a bar to represent the time in the trial
                barStart = [-0.5, -0.1]
                bar_t = t / T
                barEnd = [barStart[0] + bar_t, -0.1]
                ax.plot(
                    [barStart[0], barEnd[0]],
                    [barStart[1], barEnd[1]],
                    "b-",
                    linewidth=3,
                )
                # Plot inputs_to_env as an arrow from the hand position with length 1
                inputs_mag = torch.norm(inputs_to_env[i, t, :]).item()
                if inputs_mag > 0.001:
                    ax.arrow(
                        controlled[i, t, 0].item(),
                        controlled[i, t, 1].item(),
                        inputs_to_env[i, t, 0].item() / (50 * inputs_mag),
                        inputs_to_env[i, t, 1].item() / (50 * inputs_mag),
                        head_width=0.05,
                        head_length=0.1,
                        fc="b",
                        ec="b",
                    )

                ax.set_aspect("equal")
                # Save figure to a numpy array
                fig.canvas.draw()
                img_arr = np.array(fig.canvas.renderer.buffer_rgba())
                frames.append(img_arr)

                # Close figure to save memory
                plt.close(fig)

        # Save frames to a video using imageio
        video_path = "arm_movement_video.mp4"
        imageio.mimwrite(video_path, frames, fps=100)
        # Log the video to wandb
        wandb.log({"video": wandb.Video(video_path)})

        # Remove the video file
        os.remove(video_path)
