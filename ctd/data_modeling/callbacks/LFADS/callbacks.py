import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA

from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device

# plt.switch_backend("Agg")


def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples from the dataloaders
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        batch = next(iter(dataloader))
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=False,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = batch[0]
        # Log a few example outputs for each session
        # Convert everything to numpy
        encod_data = batch.encod_data.detach().cpu().numpy()
        recon_data = batch.recon_data.detach().cpu().numpy()
        truth = batch.truth.detach().cpu().numpy()
        means = output.output_params.detach().cpu().numpy()
        inputs = output.gen_inputs.detach().cpu().numpy()
        # Compute data sizes
        _, steps_encod, neur_encod = encod_data.shape
        _, steps_recon, neur_recon = recon_data.shape
        # Decide on how to plot panels
        if np.all(np.isnan(truth)):
            plot_arrays = [recon_data, means, inputs]
            height_ratios = [3, 3, 1]
        else:
            plot_arrays = [recon_data, truth, means, inputs]
            height_ratios = [3, 3, 3, 1]
        # Create subplots
        fig, axes = plt.subplots(
            len(plot_arrays),
            self.n_samples,
            sharex=True,
            sharey="row",
            figsize=(3 * self.n_samples, 10),
            gridspec_kw={"height_ratios": height_ratios},
        )
        for i, ax_col in enumerate(axes.T):
            for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                if j < len(plot_arrays) - 1:
                    ax.imshow(array[i].T, interpolation="none", aspect="auto")
                    ax.vlines(steps_encod, 0, neur_recon, color="orange")
                    ax.hlines(neur_encod, 0, steps_recon, color="orange")
                    ax.set_xlim(0, steps_recon)
                    ax.set_ylim(0, neur_recon)
                else:
                    ax.plot(array[i])
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            f"{self.split}/raster_plot",
            fig,
            trainer.global_step,
        )


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        dataloader = trainer.datamodule.train_dataloader()
        # Compute outputs and plot for one session at a time
        latents = []
        for batch in dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, pl_module.device)
            # Perform the forward pass through the model
            output = pl_module.predict_step(batch, None, sample_posteriors=False)
            latents.append(output.gen_states)
        latents = torch.cat(latents).detach().cpu().numpy()
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents = pca.fit_transform(latents_flat)
            latents = latents.reshape(n_samp, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "trajectory_plot",
            fig,
            trainer.global_step,
        )


class CondAvgTrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        train_conds = trainer.datamodule.train_cond_idx
        dataloader = trainer.datamodule.train_dataloader()
        # Compute outputs and plot for one session at a time
        latents = []
        for batch in dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, pl_module.device)
            # Perform the forward pass through the model
            output = pl_module.predict_step(batch, None, sample_posteriors=False)
            latents.append(output.factors)
        latents = torch.cat(latents).detach().cpu().numpy()
        # Find the condition averaged trajectory
        latents_cond_avg = np.empty((0, latents.shape[1], latents.shape[2]))
        num_conds = len(np.unique(train_conds))
        for cond in np.unique(train_conds):
            cond_idx = np.where(train_conds == cond)[0]
            cond_latents = latents[cond_idx]
            cond_latents = cond_latents.mean(0, keepdims=True)
            latents_cond_avg = np.concatenate([latents_cond_avg, cond_latents], 0)
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents_cond_avg.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents_cond_avg = pca.fit_transform(latents_flat)
            latents_cond_avg = latents_cond_avg.reshape(num_conds, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents_cond_avg:
            ax.plot(*traj.T, alpha=1, linewidth=2)
        ax.scatter(*latents_cond_avg[:, 0, :].T, alpha=1, s=20, c="g")
        ax.scatter(*latents_cond_avg[:, -1, :].T, alpha=1, s=20, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "trajectory_plot_cond_avg",
            fig,
            trainer.global_step,
        )
