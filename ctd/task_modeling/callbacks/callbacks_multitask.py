import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

# import PCA from sklearn.decomposition
from sklearn.decomposition import PCA


def angle_diff(angle1, angle2):
    """
    Calculate the smallest angle between two angles.

    Args:
    angle1 (float): The first angle in degrees.
    angle2 (float): The second angle in degrees.

    Returns:
    float: The smallest angle between angle1 and angle2 in degrees.
    """
    diff = (angle2 - angle1) % 2 * np.pi
    if diff > np.pi:
        diff -= 2 * np.pi

    return abs(diff)


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
    def __init__(self, log_every_n_epochs=100, plot_n_trials=20):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        data_dict = trainer.datamodule.all_data
        ics = data_dict["ics"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        targets = data_dict["targets"]
        inputs = data_dict["inputs"]

        # Get the first indices for each task
        included_tasks = trainer.datamodule.data_env.task_list_str

        plot_dict = {}
        for task1 in included_tasks:
            task1_inds = [i for i, task in enumerate(task_names) if task == task1]
            plot_dict[task1] = task1_inds[0]

        loggers_all = trainer.loggers
        logger = get_wandb_logger(loggers_all)

        # Pass the data through the model

        # Create plots for different cases
        fig, axes = plt.subplots(
            nrows=3,
            ncols=len(included_tasks),
            figsize=(6 * len(included_tasks), 6),
            sharex=False,
        )

        input_labels = trainer.datamodule.input_labels
        output_labels = trainer.datamodule.output_labels
        for trial_count, trial_type in enumerate(plot_dict.keys()):

            trial_num = plot_dict[trial_type]

            trial_len = phase_dict[trial_num]["response"][1]
            ax1 = axes[0][trial_count]
            ax2 = axes[1][trial_count]
            ax3 = axes[2][trial_count]
            ics_trial = ics[trial_num]
            inputs_trial = inputs[trial_num]
            # Add a batch dimension
            ics_trial = torch.Tensor(ics_trial).unsqueeze(0).to(pl_module.device)
            inputs_trial = torch.Tensor(inputs_trial).unsqueeze(0).to(pl_module.device)

            output_dict = pl_module.forward(ics_trial, inputs_trial)

            inputs_trial = inputs_trial.detach().cpu()
            controlled = output_dict["controlled"]

            task_input_str = trial_type
            pred_outputs = controlled.cpu()
            n_samples, n_timesteps, n_outputs = targets.shape

            for i in range(n_outputs):
                ax1.plot(
                    targets[trial_num, :trial_len, i],
                    label=output_labels[i],
                )

            if trial_num == 0:
                ax1.set_ylabel("Actual Outputs")
                ax2.set_ylabel("Predicted Outputs")
                ax3.set_ylabel("Inputs")
            elif trial_num == self.plot_n_trials:
                ax1.legend(loc="right")
                ax2.legend(loc="right")
                ax3.legend(loc="right")

            for i in range(n_outputs):
                ax2.plot(
                    pred_outputs[0, :trial_len, i],
                    label=output_labels[i],
                )

            _, _, n_inputs = inputs.shape
            for i in range(n_inputs):
                ax3.plot(inputs_trial[0, :trial_len, i], label=input_labels[i])
            ax1.set_title(f"Trial {trial_num}, {task_input_str}")
            ax3.set_xlabel("Time")
            plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.loggers[0].experiment.add_image(
            "state_plot", im, trainer.global_step, dataformats="HWC"
        )
        logger.log({"state_plot": wandb.Image(fig), "global_step": trainer.global_step})


class StateTransitionScatterCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, plot_n_trials=20):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        data_dict = trainer.datamodule.all_data
        ics = data_dict["ics"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        targets = data_dict["targets"]
        inputs = data_dict["inputs"]

        # Get the first indices for each task
        included_tasks = trainer.datamodule.data_env.task_list_str

        plot_dict = {}
        for task1 in included_tasks:
            task1_inds = [i for i, task in enumerate(task_names) if task == task1]
            plot_dict[task1] = task1_inds[0]

        loggers_all = trainer.loggers
        logger = get_wandb_logger(loggers_all)

        # Pass the data through the model

        # Create plots for different cases
        fig, axes = plt.subplots(
            nrows=len(included_tasks),
            ncols=4,
            figsize=(6, 2 * len(included_tasks)),
            sharex=False,
        )

        # Get color mapping for phases
        color_dict = {
            "context": "k",
            "stim1": "r",
            "mem1": "orange",
            "stim2": "b",
            "mem2": "purple",
            "response": "g",
        }

        # Plot each trial types state transitions
        for trial_count, trial_type in enumerate(plot_dict.keys()):
            trial_num = plot_dict[trial_type]
            phase_trial = phase_dict[trial_num]
            ax_inputs = axes[trial_count][0]
            ax_inputs2 = axes[trial_count][1]
            ax_outputs = axes[trial_count][2]
            ax_outputs_true = axes[trial_count][3]
            ics_trial = ics[trial_num]
            inputs_trial = inputs[trial_num]

            # Add a batch dimension
            ics_trial = torch.Tensor(ics_trial).unsqueeze(0).to(pl_module.device)
            inputs_trial = torch.Tensor(inputs_trial).unsqueeze(0).to(pl_module.device)
            targets_trial = targets[trial_num]

            output_dict = pl_module.forward(ics_trial, inputs_trial)
            inputs_trial = inputs_trial.detach().cpu()

            controlled = output_dict["controlled"]
            controlled = controlled.detach().cpu().numpy()
            for phase_count, phase in enumerate(phase_trial.keys()):
                start_ind = phase_trial[phase][0]
                end_ind = phase_trial[phase][1]
                ax_inputs.plot(
                    inputs_trial[0, start_ind:end_ind, 1],
                    inputs_trial[0, start_ind:end_ind, 2],
                    c=color_dict[phase],
                    label=phase,
                )
                ax_inputs2.plot(
                    inputs_trial[0, start_ind:end_ind, 3],
                    inputs_trial[0, start_ind:end_ind, 4],
                    c=color_dict[phase],
                    label=phase,
                )
                ax_outputs.plot(
                    controlled[0, start_ind:end_ind, 1],
                    controlled[0, start_ind:end_ind, 2],
                    c=color_dict[phase],
                    label=phase,
                )
                ax_outputs_true.scatter(
                    targets_trial[start_ind:end_ind, 1],
                    targets_trial[start_ind:end_ind, 2],
                    c=color_dict[phase],
                    label=phase,
                )

            if trial_count == 0:
                ax_inputs.set_title("Inputs 1")
                ax_inputs2.set_title("Inputs 2")
                ax_outputs.set_title("Outputs")
                ax_outputs_true.set_title("Outputs True")
            ax_inputs.set_ylabel(trial_type)
            ax_inputs.set_xticklabels([])
            ax_inputs.set_yticklabels([])
            ax_inputs2.set_xticklabels([])
            ax_inputs2.set_yticklabels([])
            ax_outputs.set_xticklabels([])
            ax_outputs.set_yticklabels([])
            ax_outputs_true.set_xticklabels([])
            ax_outputs_true.set_yticklabels([])

            ax_inputs.set_xlim(-2, 2)
            ax_inputs.set_ylim(-2, 2)
            ax_inputs2.set_xlim(-2, 2)
            ax_inputs2.set_ylim(-2, 2)
            ax_outputs.set_xlim(-2, 2)
            ax_outputs.set_ylim(-2, 2)
            ax_outputs_true.set_xlim(-2, 2)
            ax_outputs_true.set_ylim(-2, 2)

            ax_inputs.set_aspect("equal", adjustable="box")
            ax_inputs2.set_aspect("equal", adjustable="box")
            ax_outputs.set_aspect("equal", adjustable="box")
            ax_outputs_true.set_aspect("equal", adjustable="box")

            plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.loggers[0].experiment.add_image(
            "state_plot_scatter", im, trainer.global_step, dataformats="HWC"
        )
        logger.log(
            {"state_plot_scatter": wandb.Image(fig), "global_step": trainer.global_step}
        )


class MultiTaskPerformanceCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, plot_n_trials=20):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):
        """Computes the performance of the models based on the
        angle difference between the output and the target.

        For each task, the angle difference is computed for the
        last 1/4 of the response period.

        The percent success is computed as the percentage of
        trials where the angle difference is less than pi/10.

        The percent success is logged to wandb.

        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        #

        # Get the data from the datamodule
        data_dict = trainer.datamodule.all_data
        ics = data_dict["ics"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        targets = data_dict["targets"]
        inputs = data_dict["inputs"]

        logger = get_wandb_logger(trainer.loggers)
        percent_success = np.zeros(len(trainer.datamodule.data_env.task_list_str))

        # find indices where task_to_analyze is the task
        for task_num, task_to_analyze in enumerate(
            trainer.datamodule.data_env.task_list_str
        ):
            task_inds = [
                i for i, task in enumerate(task_names) if task == task_to_analyze
            ]

            # Get the inputs, ics, and phase_dict for the task
            task_inputs = torch.Tensor(inputs[task_inds]).to(pl_module.device)
            task_ics = torch.Tensor(ics[task_inds]).to(pl_module.device)
            task_targets = torch.Tensor(targets[task_inds])
            task_phase_dict = [
                dict1 for i, dict1 in enumerate(phase_dict) if i in task_inds
            ]

            # Pass data through the model
            output_dict = pl_module.forward(task_ics, task_inputs)
            tt_outputs = output_dict["controlled"]

            tt_outputs = tt_outputs.detach().cpu()
            task_targets = task_targets.detach().cpu()

            perf = np.zeros(len(task_phase_dict))
            perf_dist = np.ones(len(task_phase_dict))
            no_respond_trial = np.zeros(len(task_phase_dict), dtype=bool)

            # Iterate through task trials
            for i in range(len(task_phase_dict)):
                response_edges = task_phase_dict[i]["response"]
                response_len = response_edges[1] - response_edges[0]

                # Compute average angle for the last 1/4 of the response period
                response_val = tt_outputs[
                    i,
                    response_edges[0] + (3 * response_len // 4) : response_edges[1],
                    1:,
                ]
                mean_response = torch.mean(response_val, dim=0)
                mean_angle = torch.atan2(mean_response[1], mean_response[0])

                # Get the target angle
                response_target = task_targets[
                    i, response_edges[0] + response_len // 2 : response_edges[1], 1:
                ]
                mean_target = torch.mean(response_target, dim=0)
                mean_target_angle = torch.atan2(mean_target[1], mean_target[0])

                # Compute the performance (angle difference)
                perf[i] = angle_diff(mean_angle, mean_target_angle)

                # if no response was correct, check if the response was close to 0
                if torch.sum(np.abs(mean_target)) == 0:
                    no_respond_trial[i] = True
                    perf_dist[i] = torch.sum(mean_response)

            # Compute the percent success (angle < pi/10)
            flag_success_angle = perf < (np.pi / 10)
            flag_success_noreport = np.logical_and(perf_dist < 0.1, no_respond_trial)
            flag_success = np.logical_or(flag_success_angle, flag_success_noreport)
            percent_success = np.sum(flag_success) / len(flag_success)

            # Log percent success to wandb
            logger.log(
                {
                    f"success_rate/{task_to_analyze}": percent_success,
                    "global_step": trainer.global_step,
                }
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
        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        output_dict = pl_module.forward(ics, inputs)
        trajs_out = output_dict["controlled"]
        logger = get_wandb_logger(trainer.loggers)

        # Plot the true and predicted trajectories
        trial_vec = torch.tensor(
            np.random.randint(0, trajs_out.shape[0], self.num_trials_to_plot)
        )
        fig, ax = plt.subplots()
        traj_in = 1  # TODO: Fix this
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
        logger.add_image(
            "trajectory_plot_over_time", im, trainer.global_step, dataformats="HWC"
        )


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
        output_dict = pl_module.forward(ics_train, inputs_train)

        lats_train = output_dict["latents"]
        lats_train = lats_train.detach().cpu().numpy()
        n_trials, n_times, n_lat_dim = lats_train.shape
        if n_lat_dim > 3:
            pca1 = PCA(n_components=3)
            lats_train = pca1.fit_transform(lats_train.reshape(-1, n_lat_dim))
            lats_train = lats_train.reshape(n_trials, n_times, 3)
            exp_var = np.sum(pca1.explained_variance_ratio_)
        else:
            exp_var = 1.0

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
