import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import FastICA

# import linear regression
from sklearn.linear_model import LinearRegression

from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device


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


class InputR2Plot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(
        self, split="valid", log_every_n_epochs=100, n_samples=16, use_ica=True
    ):
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
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = n_samples
        self.use_ica = use_ica

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
        extra = batch[1][0]
        batch = batch[0]
        n_batch, n_time, n_inf_inputs = output.gen_inputs.shape
        n_true_inputs = extra.shape[-1]

        # Convert everything to numpy
        encod_data = batch.encod_data.detach().cpu().numpy()
        recon_data = batch.recon_data.detach().cpu().numpy()
        means = output.output_params.detach().cpu().numpy()
        inf_inputs = output.gen_inputs.detach().cpu().numpy()
        # perform ICA on inferred inputs
        if self.use_ica:
            ica = FastICA(n_components=n_inf_inputs)
            inf_inputs = ica.fit_transform(inf_inputs.reshape(-1, n_inf_inputs))
            inf_inputs = inf_inputs.reshape(-1, n_time, n_inf_inputs)
        true_inputs = extra.detach().cpu().numpy()
        # true_inputs = batch[s].ext_input.detach().cpu().numpy()
        # Compute data sizes
        _, steps_encod, neur_encod = encod_data.shape
        _, steps_recon, neur_recon = recon_data.shape

        # Fit lin model from inferred inputs to true inputs
        lr = LinearRegression().fit(
            inf_inputs.reshape(-1, n_inf_inputs),
            true_inputs.reshape(-1, n_true_inputs),
        )
        # Get R2
        r2_inf_to_true = lr.score(
            inf_inputs.reshape(-1, n_inf_inputs),
            true_inputs.reshape(-1, n_true_inputs),
        )
        # Fit lin model from true inputs to inferred inputs
        lr2 = LinearRegression().fit(
            true_inputs.reshape(-1, n_true_inputs),
            inf_inputs.reshape(-1, n_inf_inputs),
        )

        inf_inputs = lr.predict(inf_inputs.reshape(-1, n_inf_inputs)).reshape(
            -1, n_time, n_true_inputs
        )
        inf_inputs = inf_inputs.reshape(-1, n_time, n_true_inputs)

        # Get R2
        r2_true_to_inf = lr2.score(
            true_inputs.reshape(-1, n_true_inputs),
            inf_inputs.reshape(-1, n_inf_inputs),
        )

        metrics = {
            "input_R2/r2_inf_to_true": r2_inf_to_true,
            "input_R2/r2_true_to_inf": r2_true_to_inf,
        }
        # Log the figure
        pl_module.log_dict(
            {
                **metrics,
            }
        )
        labels = ["ReconFR", "InferredFR", "TrueInputs", "InfInputs"]
        plot_arrays = [recon_data, means, true_inputs, inf_inputs]
        height_ratios = [3, 3, 1, 1]
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
                if j < len(plot_arrays) - 2:
                    ax.imshow(array[i].T, interpolation="none", aspect="auto")
                    ax.vlines(steps_encod, 0, neur_recon, color="orange")
                    ax.hlines(neur_encod, 0, steps_recon, color="orange")
                    ax.set_xlim(0, steps_recon)
                    ax.set_ylim(0, neur_recon)
                else:
                    ax.plot(array[i])
                if i == 0:
                    ax.set_ylabel(labels[j])
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            f"{self.split}/raster_plot_inputs",
            fig,
            trainer.global_step,
        )

        # Generate figure


class CondAvgRaster(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", log_every_n_epochs=100):
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

        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        train_conds = trainer.datamodule.train_cond_idx
        dataloaders = {s: dls["train"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            true_rates = []
            inf_rates = []
            inputs = []
            for batch in dataloader:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                true_rates.append(batch[0][s].recon_data)
                inf_rates.append(output.output_params)
                inputs.append(output.gen_inputs)
            true_rates = torch.cat(true_rates).detach().cpu().numpy()
            inf_rates = torch.cat(inf_rates).detach().cpu().numpy()
            inputs = torch.cat(inputs).detach().cpu().numpy()

            # Find the condition averaged trajectory
            true_cond_avg = np.empty((0, true_rates.shape[1], true_rates.shape[2]))
            inf_cond_avg = np.empty((0, inf_rates.shape[1], inf_rates.shape[2]))
            inputs_cond_avg = np.empty((0, inputs.shape[1], inputs.shape[2]))

            condList = trainer.datamodule.condition_list
            condList[:, :2] = np.round(
                condList[:, :2] - np.mean(condList[:, :2], axis=0), 5
            )
            maxForce = np.max(np.abs(condList[:, 2:]))
            condList[:, 2:] = np.round(condList[:, 2:] / (20 * maxForce), 5)

            num_conds = len(np.unique(train_conds))
            for cond in np.unique(train_conds):
                cond_idx = np.where(train_conds == cond)[0]
                cond_true = true_rates[cond_idx]
                cond_inf = inf_rates[cond_idx]
                cond_inputs = inputs[cond_idx]

                cond_true = cond_true.mean(0, keepdims=True)
                cond_inf = cond_inf.mean(0, keepdims=True)
                cond_inputs = cond_inputs.mean(0, keepdims=True)

                true_cond_avg = np.concatenate([true_cond_avg, cond_true], 0)
                inf_cond_avg = np.concatenate([inf_cond_avg, cond_inf], 0)
                inputs_cond_avg = np.concatenate([inputs_cond_avg, cond_inputs], 0)
            # Subtract mean from inputs cond avg
            inputs_cond_avg = inputs_cond_avg - inputs_cond_avg.mean(0, keepdims=True)

            _, steps_encod, neur_encod = true_cond_avg.shape
            _, steps_recon, neur_recon = inf_cond_avg.shape
            inputs_cond_avg = inputs_cond_avg.reshape(-1, inputs_cond_avg.shape[-1])
            ica = FastICA(n_components=inputs_cond_avg.shape[-1])
            ica_inputs = ica.fit_transform(inputs_cond_avg)
            inputs_cond_avg = ica_inputs.reshape(-1, steps_encod, ica_inputs.shape[-1])

            # Decide on how to plot panels
            plot_arrays = [None, true_cond_avg, inf_cond_avg, inputs_cond_avg]
            plot_titles = [None, "True Rates", "Inferred Rates", "Inputs"]
            height_ratios = [3, 3, 3, 1]
            # Create subplots
            fig, axes = plt.subplots(
                len(plot_arrays),
                num_conds,
                sharex=False,
                sharey="row",
                figsize=(3 * num_conds, 10),
                gridspec_kw={"height_ratios": height_ratios},
            )
            # Plot the location of the target and direction of force
            for i, ax_col in enumerate(axes.T):
                for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                    if j == 0:
                        # Scatter plot of target location
                        ax.scatter(
                            condList[i, 0],
                            condList[i, 1],
                            color="r",
                            s=100,
                        )
                        # Plot arrow for cond 2, 3
                        ax.arrow(
                            0,
                            0,
                            condList[i, 2],
                            condList[i, 3],
                        )
                        ax.set_xlim(-0.15, 0.15)
                        ax.set_ylim(-0.15, 0.15)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    elif j < len(plot_arrays) - 1:
                        ax.imshow(array[i].T, interpolation="none", aspect="auto")
                        ax.vlines(steps_encod, 0, neur_recon, color="orange")
                        ax.hlines(neur_encod, 0, steps_recon, color="orange")
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylim(0, neur_recon)
                        ax.set_ylabel(plot_titles[j])
                    else:
                        ax.plot(array[i])
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylabel(plot_titles[j])

            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"{self.split}/raster_plot_cond_avg/sess{s}",
                fig,
                trainer.global_step,
            )
