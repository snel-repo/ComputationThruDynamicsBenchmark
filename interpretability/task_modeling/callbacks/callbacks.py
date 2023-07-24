import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from sklearn.model_selection import train_test_split
import h5py
# import PCA from sklearn.decomposition
from sklearn.decomposition import PCA

# plt.switch_backend("Agg")
DATA_HOME = "/home/csverst/Documents/tempData/"


def sigmoidActivation(module, input):
    return 1 / (1 + module.exp(-1 * input))

def apply_data_warp_sigmoid(data):
    warp_functions = [sigmoidActivation, sigmoidActivation, sigmoidActivation]
    firingMax = [2, 2, 2, 2]
    numDims = data.shape[1]

    a = np.array(1)
    dataGen = type(a) == type(data)
    if dataGen:
        module = np
    else:
        module = torch

    for i in range(numDims):

        j = np.mod(i, len(warp_functions) * len(firingMax))
        # print(f'Max firing {firingMax[np.mod(j, len(firingMax))]}
        # warp {warp_functions[int(np.floor((j)/(len(warp_functions)+1)))]}')
        data[:, i] = firingMax[np.mod(j, len(firingMax))] * warp_functions[
            int(np.floor((j) / (len(warp_functions) + 1)))
        ](module, data[:, i])

    return data

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

class StateTransitionCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, plot_n_trials = 5):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        outputs = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        
        logger = trainer.loggers[2].experiment
        # Pass the data through the model
        pred_outputs, lats = pl_module.forward(inputs)
        # Create plots for different cases
        fig, axes = plt.subplots(nrows = 3, ncols = self.plot_n_trials, figsize=(8*self.plot_n_trials, 6),sharex=True)
        for trial_num in range(self.plot_n_trials):
            ax1 = axes[0][trial_num]
            ax2 = axes[1][trial_num]
            ax3 = axes[2][trial_num]
            
            outputs = outputs.cpu()
            inputs = inputs.cpu()
            pred_outputs = pred_outputs.cpu()
            n_samples, n_timesteps, n_outputs = outputs.shape
            input_labels = trainer.datamodule.input_labels
            output_labels = trainer.datamodule.output_labels

            for i in range(n_outputs):
                ax1.plot(outputs[trial_num,:,i], label = output_labels[i])
            ax1.legend(loc = 'right')
            ax1.set_ylabel("Actual Outputs")
            
            for i in range(n_outputs):
                ax2.plot(pred_outputs[trial_num,:,i], label = output_labels[i])
            ax2.set_ylabel("Predicted Outputs")
            ax2.legend(loc = 'right')

            _,_, n_inputs = inputs.shape
            for i in range(n_inputs):
                ax3.plot(inputs[trial_num,:,i], label = input_labels[i])
            
            ax3.legend(loc = 'right')
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Inputs")
            
            plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.loggers[0].experiment.add_image(
            "state_plot", im, trainer.global_step, dataformats="HWC"
        )
        logger.log(
            {"state_plot": wandb.Image(fig), "global_step": trainer.global_step}
        )

class TrajectoryPlotOverTimeCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=5, axis_num=0):

        self.log_every_n_epochs = log_every_n_epochs
        self.num_trials_to_plot = num_trials_to_plot
        self.axis_num = axis_num

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        trajs_out, _ = pl_module.forward(inputs)
        # Plot the true and predicted trajectories
        trial_vec = torch.tensor(
            np.random.randint(0, trajs_out.shape[0], self.num_trials_to_plot)
        )
        fig, ax = plt.subplots()
        traj_in = data
        t1 = np.linspace(0, 1, len(trial_vec) * trajs_out.shape[1])

        def prep_trajs(x):
            return x[trial_vec, :, self.axis_num].detach().cpu().numpy().flatten()

        ax.plot(t1, prep_trajs(traj_in), "C0", label="Actual Traj")
        ax.plot(t1, np.exp(prep_trajs(trajs_out)), "C1", label="Pred Traj")
        ax.set_xlabel("Time (AU)")
        ax.set_ylabel("Firing (AU)")
        ax.set_title(f"axis {self.axis_num}, {self.num_trials_to_plot} trials")
        ax.legend()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.logger.experiment[0].add_image(
            "trajectory_plot_over_time", im, trainer.global_step, dataformats="HWC"
        )


class LatentTrajectoryPlot(pl.Callback):
    def __init__(self, log_every_n_epochs=10, 
                 ):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = trainer.loggers[2].experiment

        # Get trajectories and model predictions
        train_dataloader = trainer.datamodule.train_dataloader()
        inputs_train = torch.cat([batch[0] for batch in train_dataloader]).to(pl_module.device)
        _, lats_train = pl_module.forward(inputs_train)
        
        lats_train = lats_train.detach().cpu().numpy()
        n_trials, n_times, n_lat_dim = lats_train.shape
        if n_lat_dim >3:
            pca1= PCA(n_components=3)
            lats_train = pca1.fit_transform(lats_train.reshape(-1, n_lat_dim))
            lats_train = lats_train.reshape(n_trials, n_times, 3)
            exp_var = np.sum(pca1.explained_variance_ratio_)
        else:
            exp_var = 1.0
        
        # Plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for traj in lats_train:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*lats_train[:,0,:].T, alpha=0.1, s=10, c ='g')
        ax.scatter(*lats_train[:,-1,:].T, alpha=0.1, s=10, c ='r')
        ax.set_title(f"explained variance: {exp_var:.2f}")
        plt.tight_layout()
        trainer.loggers[0].experiment.add_figure("latent_trajectory", fig, global_step=trainer.global_step)
        logger.log(
            {"latent_traj": wandb.Image(fig), "global_step": trainer.global_step}
        )
