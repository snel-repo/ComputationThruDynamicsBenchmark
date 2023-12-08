import io
import os

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

# import PCA from sklearn.decomposition
from sklearn.decomposition import PCA

# plt.switch_backend("Agg")
DATA_HOME = "/home/csverst/Documents/tempData/"


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

        logger = trainer.loggers[2].experiment

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

        controlled, lats_train, actions = pl_module.forward(
            ics_train, inputs_train, targets_train
        )

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
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=5):

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

        controlled, latents, actions = pl_module.forward(ics, inputs, targets)

        B, T, N = controlled.shape

        # Create a list to store frames
        frames = []

        # Prepare to log a video for all batches
        for i in range(self.num_trials_to_plot):
            for t in range(T):
                # Create a figure
                fig, ax = plt.subplots(figsize=(3, 3))

                # Plot target position as a red square
                target = patches.Rectangle(
                    (targets[i, t, 0].item() - 0.05, targets[i, t, 1].item() - 0.05),
                    0.1,
                    0.1,
                    color="red",
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

                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

                ax.set_xticks([])
                ax.set_yticks([])

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
