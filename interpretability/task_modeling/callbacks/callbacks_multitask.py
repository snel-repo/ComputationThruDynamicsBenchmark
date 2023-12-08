import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

# import PCA from sklearn.decomposition
from sklearn.decomposition import PCA

# plt.switch_backend("Agg")
DATA_HOME = "/home/csverst/Documents/tempData/"


def angle_diff(angle1, angle2):
    """
    Calculate the smallest angle between two angles.

    Args:
    angle1 (float): The first angle in degrees.
    angle2 (float): The second angle in degrees.

    Returns:
    float: The smallest angle between angle1 and angle2 in degrees.
    """
    # Calculate the difference in angles and take modulo 360
    diff = (angle2 - angle1) % 2 * np.pi

    # Adjust the difference to ensure it's the smallest angle
    if diff > np.pi:
        diff -= 2 * np.pi

    return abs(diff)


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
    def __init__(self, log_every_n_epochs=100, plot_n_trials=20):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        targets = torch.cat([batch[2] for batch in dataloader]).to(pl_module.device)

        logger = trainer.loggers[2].experiment
        # Pass the data through the model
        controlled, lats, _ = pl_module.forward(ics, inputs)
        # Create plots for different cases
        fig, axes = plt.subplots(
            nrows=3,
            ncols=self.plot_n_trials,
            figsize=(6 * self.plot_n_trials, 6),
            sharex=False,
        )

        input_labels = trainer.datamodule.input_labels
        output_labels = trainer.datamodule.output_labels
        task_list_str = trainer.datamodule.data_env.task_list_str
        mask = (inputs.sum(dim=2, keepdim=True) != 0).float()
        trial_lens = mask.sum(dim=1).squeeze().cpu().numpy().astype(int)
        for trial_num in range(self.plot_n_trials):
            ax1 = axes[0][trial_num]
            ax2 = axes[1][trial_num]
            ax3 = axes[2][trial_num]
            targets = targets.cpu()
            inputs = inputs.cpu()
            task_input = inputs[trial_num, 0, 5:]
            task_input_str = task_list_str[task_input.argmax()]
            pred_outputs = controlled.cpu()
            n_samples, n_timesteps, n_outputs = targets.shape

            for i in range(n_outputs):
                ax1.plot(
                    targets[trial_num, : trial_lens[trial_num], i],
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
                    pred_outputs[trial_num, : trial_lens[trial_num], i],
                    label=output_labels[i],
                )

            _, _, n_inputs = inputs.shape
            for i in range(n_inputs):
                ax3.plot(
                    inputs[trial_num, : trial_lens[trial_num], i], label=input_labels[i]
                )
            ax1.set_title(f"Trial {trial_num}, {task_input_str}")
            ax3.set_xlabel("Time")
            plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.loggers[0].experiment.add_image(
            "state_plot", im, trainer.global_step, dataformats="HWC"
        )
        logger.log({"state_plot": wandb.Image(fig), "global_step": trainer.global_step})


class MultiTaskPerformanceCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, plot_n_trials=20):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions

        # Get the data from the datamodule
        data_dict = trainer.datamodule.all_data
        ics = data_dict["ics"]
        phase_dict = data_dict["phase_dict"]
        task_names = data_dict["task_names"]
        targets = data_dict["targets"]
        inputs = data_dict["inputs"]

        logger = trainer.loggers[2].experiment
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
            tt_outputs, tt_latents, _ = pl_module.forward(task_ics, task_inputs)
            tt_outputs = tt_outputs.detach().cpu()
            task_targets = task_targets.detach().cpu()

            perf = np.zeros(len(task_phase_dict))
            for i in range(len(task_phase_dict)):
                response_edges = task_phase_dict[i]["response"]
                response_len = response_edges[1] - response_edges[0]
                response_val = tt_outputs[
                    i, response_edges[0] + response_len // 2 : response_edges[1], 1:
                ]
                mean_response = torch.mean(response_val, dim=0)
                mean_angle = torch.atan2(mean_response[1], mean_response[0])
                response_target = task_targets[
                    i, response_edges[0] + response_len // 2 : response_edges[1], 1:
                ]
                mean_target = torch.mean(response_target, dim=0)
                mean_target_angle = torch.atan2(mean_target[1], mean_target[0])
                perf[i] = angle_diff(mean_angle, mean_target_angle)
            percent_success = np.sum(perf < (np.pi / 10)) / len(perf)
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
        trajs_out, _, _ = pl_module.forward(ics, inputs)
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
        trainer.logger.experiment[0].add_image(
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

        logger = trainer.loggers[2].experiment

        # Get trajectories and model predictions
        train_dataloader = trainer.datamodule.train_dataloader()
        ics_train = torch.cat([batch[0] for batch in train_dataloader]).to(
            pl_module.device
        )
        inputs_train = torch.cat([batch[1] for batch in train_dataloader]).to(
            pl_module.device
        )
        _, lats_train, _ = pl_module.forward(ics_train, inputs_train)

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


class SimonSaysCondAvgLats(pl.Callback):
    def __init__(
        self,
        log_every_n_epochs=10,
    ):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        logger = trainer.loggers[2].experiment
        task_env = pl_module.task_env

        targets_fwd = np.array([0, 1, 2])
        queue_input_fwd, queue_output_fwd = task_env.generate_dataset(
            n_samples=100, targets=targets_fwd, isFIFO=1
        )
        stack_input_fwd, stack_output_fwd = task_env.generate_dataset(
            n_samples=100, targets=targets_fwd, isFIFO=-1
        )

        targets_rev = np.array([2, 1, 0])
        queue_input_rev, queue_output_rev = task_env.generate_dataset(
            n_samples=100, targets=targets_rev, isFIFO=1
        )
        stack_input_rev, stack_output_rev = task_env.generate_dataset(
            n_samples=100, targets=targets_rev, isFIFO=-1
        )

        # Get trajectories and model predictions

        _, lats_queue_fwd = pl_module.forward(
            torch.Tensor(queue_input_fwd).to(pl_module.device)
        )
        _, lats_stack_fwd = pl_module.forward(
            torch.Tensor(stack_input_fwd).to(pl_module.device)
        )
        _, lats_queue_rev = pl_module.forward(
            torch.Tensor(queue_input_rev).to(pl_module.device)
        )
        _, lats_stack_rev = pl_module.forward(
            torch.Tensor(stack_input_rev).to(pl_module.device)
        )

        delay_inp = 2
        go_inp = 3

        lats_queue_fwd = lats_queue_fwd.detach().cpu().numpy()
        lats_stack_fwd = lats_stack_fwd.detach().cpu().numpy()
        lats_queue_rev = lats_queue_rev.detach().cpu().numpy()
        lats_stack_rev = lats_stack_rev.detach().cpu().numpy()

        # Align the lats to when the delay turns on.
        # The first index per trial where input[:,:,delay_inp] > 0.5
        delay_lats_queue_fwd = np.zeros(
            (lats_queue_fwd.shape[0], 20, lats_queue_fwd.shape[2])
        )
        delay_lats_stack_fwd = np.zeros(
            (lats_stack_fwd.shape[0], 20, lats_stack_fwd.shape[2])
        )
        delay_lats_queue_rev = np.zeros(
            (lats_queue_rev.shape[0], 20, lats_queue_rev.shape[2])
        )
        delay_lats_stack_rev = np.zeros(
            (lats_stack_rev.shape[0], 20, lats_stack_rev.shape[2])
        )

        for i in range(lats_queue_fwd.shape[0]):
            delay_ind = np.where(queue_input_fwd[i, :, delay_inp] > 0.5)[0][0]
            delay_lats_queue_fwd[i, :, :] = lats_queue_fwd[
                i, delay_ind - 10 : delay_ind + 10, :
            ]
            delay_ind = np.where(stack_input_fwd[i, :, delay_inp] > 0.5)[0][0]
            delay_lats_stack_fwd[i, :, :] = lats_stack_fwd[
                i, delay_ind - 10 : delay_ind + 10, :
            ]
            delay_ind = np.where(queue_input_rev[i, :, delay_inp] > 0.5)[0][0]
            delay_lats_queue_rev[i, :, :] = lats_queue_rev[
                i, delay_ind - 10 : delay_ind + 10, :
            ]
            delay_ind = np.where(stack_input_rev[i, :, delay_inp] > 0.5)[0][0]
            delay_lats_stack_rev[i, :, :] = lats_stack_rev[
                i, delay_ind - 10 : delay_ind + 10, :
            ]

        # Align to when the go signal turns on (input[:,:,go_inp] > 0.5)
        go_lats_queue_fwd = np.zeros(
            (lats_queue_fwd.shape[0], 20, lats_queue_fwd.shape[2])
        )
        go_lats_stack_fwd = np.zeros(
            (lats_stack_fwd.shape[0], 20, lats_stack_fwd.shape[2])
        )
        go_lats_queue_rev = np.zeros(
            (lats_queue_rev.shape[0], 20, lats_queue_rev.shape[2])
        )
        go_lats_stack_rev = np.zeros(
            (lats_stack_rev.shape[0], 20, lats_stack_rev.shape[2])
        )

        for i in range(lats_queue_fwd.shape[0]):
            go_ind = np.where(queue_input_fwd[i, :, go_inp] > 0.5)[0][0]
            go_lats_queue_fwd[i, :, :] = lats_queue_fwd[i, go_ind - 10 : go_ind + 10, :]
            go_ind = np.where(stack_input_fwd[i, :, go_inp] > 0.5)[0][0]
            go_lats_stack_fwd[i, :, :] = lats_stack_fwd[i, go_ind - 10 : go_ind + 10, :]
            go_ind = np.where(queue_input_rev[i, :, go_inp] > 0.5)[0][0]
            go_lats_queue_rev[i, :, :] = lats_queue_rev[i, go_ind - 10 : go_ind + 10, :]
            go_ind = np.where(stack_input_rev[i, :, go_inp] > 0.5)[0][0]
            go_lats_stack_rev[i, :, :] = lats_stack_rev[i, go_ind - 10 : go_ind + 10, :]

        # Average across trials
        delay_lats_queue_fwd_mean = np.mean(delay_lats_queue_fwd, axis=0)
        delay_lats_stack_fwd_mean = np.mean(delay_lats_stack_fwd, axis=0)
        delay_lats_queue_rev_mean = np.mean(delay_lats_queue_rev, axis=0)
        delay_lats_stack_rev_mean = np.mean(delay_lats_stack_rev, axis=0)

        go_lats_queue_fwd_mean = np.mean(go_lats_queue_fwd, axis=0)
        go_lats_stack_fwd_mean = np.mean(go_lats_stack_fwd, axis=0)
        go_lats_queue_rev_mean = np.mean(go_lats_queue_rev, axis=0)
        go_lats_stack_rev_mean = np.mean(go_lats_stack_rev, axis=0)

        mean_delay_lats = np.stack(
            (
                delay_lats_queue_fwd_mean,
                delay_lats_stack_fwd_mean,
                delay_lats_queue_rev_mean,
                delay_lats_stack_rev_mean,
            )
        )

        mean_go_lats = np.stack(
            (
                go_lats_queue_fwd_mean,
                go_lats_stack_fwd_mean,
                go_lats_queue_rev_mean,
                go_lats_stack_rev_mean,
            )
        )

        labels = ["Queue Fwd", "Stack Fwd", "Queue Rev", "Stack Rev"]
        # Combine the delay lats
        delay_lats = np.vstack(
            (
                delay_lats_queue_fwd,
                delay_lats_stack_fwd,
                delay_lats_queue_rev,
                delay_lats_stack_rev,
            )
        )

        go_lats = np.vstack(
            (go_lats_queue_fwd, go_lats_stack_fwd, go_lats_queue_rev, go_lats_stack_rev)
        )

        n_trials, n_times, n_lat_dim = delay_lats.shape
        if n_lat_dim > 3:
            pcaDelay = PCA(n_components=3)
            delay_lats = pcaDelay.fit_transform(delay_lats.reshape(-1, n_lat_dim))
            delay_lats = delay_lats.reshape(n_trials, n_times, 3)
            mean_delay_lats = pcaDelay.transform(mean_delay_lats.reshape(-1, n_lat_dim))
            mean_delay_lats = mean_delay_lats.reshape(4, n_times, 3)

            exp_var_delay = np.sum(pcaDelay.explained_variance_ratio_)
            pcaMove = PCA(n_components=3)
            go_lats = pcaMove.fit_transform(go_lats.reshape(-1, n_lat_dim))
            go_lats = go_lats.reshape(n_trials, n_times, 3)
            mean_go_lats = pcaMove.transform(mean_go_lats.reshape(-1, n_lat_dim))
            mean_go_lats = mean_go_lats.reshape(4, n_times, 3)

            exp_var_go = np.sum(pcaMove.explained_variance_ratio_)

        else:
            exp_var_delay = 1.0
            exp_var_go = 1.0

        # Plot trajectories
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(211, projection="3d")
        for traj in delay_lats:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        for i, traj in enumerate(mean_delay_lats):
            ax.plot(*traj.T, alpha=1, linewidth=2, label=labels[i])

        ax.scatter(*delay_lats[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*delay_lats[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.legend()
        ax.set_title(f"Delay explained variance: {exp_var_delay:.2f}")
        plt.tight_layout()
        ax = fig.add_subplot(212, projection="3d")
        for traj in go_lats:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        for i, traj in enumerate(mean_go_lats):
            ax.plot(*traj.T, alpha=1, linewidth=2, label=labels[i])
        ax.scatter(*go_lats[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*go_lats[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"Go explained variance: {exp_var_go:.2f}")
        ax.legend()
        plt.tight_layout()

        trainer.loggers[0].experiment.add_figure(
            "delay_go_latents", fig, global_step=trainer.global_step
        )
        logger.log(
            {"delay_go_latents": wandb.Image(fig), "global_step": trainer.global_step}
        )


class SimonSaysLatICs(pl.Callback):
    def __init__(
        self,
        log_every_n_epochs=10,
    ):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = trainer.loggers[2].experiment
        task_env = pl_module.task_env
        target_lists = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]
        target_labels = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
        n_samples = 100
        n_lats = pl_module.model.latent_size

        queue_input_vecs = np.zeros((len(target_lists), n_samples, 180, 5))
        stack_input_vecs = np.zeros((len(target_lists), n_samples, 180, 5))
        queue_lats = np.zeros((len(target_lists), n_samples, 180, n_lats))
        stack_lats = np.zeros((len(target_lists), n_samples, 180, n_lats))

        delay_inp = 2
        go_inp = 3

        for i, target_list in enumerate(target_lists):
            queue_input_vecs[i, :, :, :], _ = task_env.generate_dataset(
                n_samples=n_samples, targets=target_list, isFIFO=1
            )
            stack_input_vecs[i, :, :, :], _ = task_env.generate_dataset(
                n_samples=n_samples, targets=target_list, isFIFO=-1
            )
            queue_lats[i, :, :, :] = (
                pl_module.forward(
                    torch.Tensor(queue_input_vecs[i, :, :]).to(pl_module.device)
                )[1]
                .detach()
                .cpu()
                .numpy()
            )
            stack_lats[i, :, :, :] = (
                pl_module.forward(
                    torch.Tensor(stack_input_vecs[i, :, :]).to(pl_module.device)
                )[1]
                .detach()
                .cpu()
                .numpy()
            )

        # Get trajectories and model predictions
        n_lats = queue_lats.shape[-1]

        queue_lats_delay = np.zeros((len(target_lists), n_samples, 20, n_lats))
        stack_lats_delay = np.zeros((len(target_lists), n_samples, 20, n_lats))

        for i in range(queue_lats.shape[0]):
            for j in range(queue_input_vecs.shape[1]):
                delay_ind = np.where(queue_input_vecs[i, j, :, delay_inp] > 0.5)[0][0]
                queue_lats_delay[i, j, :, :] = queue_lats[
                    i, j, delay_ind - 10 : delay_ind + 10, :
                ]
                delay_ind = np.where(stack_input_vecs[i, j, :, delay_inp] > 0.5)[0][0]
                stack_lats_delay[i, j, :, :] = stack_lats[
                    i, j, delay_ind - 10 : delay_ind + 10, :
                ]

        queue_lats_go = np.zeros((len(target_lists), n_samples, 20, n_lats))
        stack_lats_go = np.zeros((len(target_lists), n_samples, 20, n_lats))

        for i in range(queue_lats.shape[0]):
            for j in range(queue_input_vecs.shape[1]):
                go_ind = np.where(queue_input_vecs[i, j, :, go_inp] > 0.5)[0][0]
                queue_lats_go[i, j, :, :] = queue_lats[
                    i, j, go_ind - 10 : go_ind + 10, :
                ]
                go_ind = np.where(stack_input_vecs[i, j, :, go_inp] > 0.5)[0][0]
                stack_lats_go[i, j, :, :] = stack_lats[
                    i, j, go_ind - 10 : go_ind + 10, :
                ]

        # Get all delay ICs
        delay_ics_stack = stack_lats_delay[:, :, 0, :]
        delay_ics_queue = queue_lats_delay[:, :, 0, :]
        delay_ics = np.vstack((delay_ics_queue, delay_ics_stack))
        delay_ics_flat = delay_ics.reshape(-1, delay_ics.shape[-1])

        # Get all go ICs
        go_ics_stack = stack_lats_go[:, :, 0, :]
        go_ics_queue = queue_lats_go[:, :, 0, :]
        go_ics = np.vstack((go_ics_queue, go_ics_stack))
        go_ics_flat = go_ics.reshape(-1, go_ics.shape[-1])

        n_seq, n_trials, n_lat_dim = delay_ics.shape
        if n_lat_dim > 3:
            pcaDelay = PCA(n_components=3)
            delay_ics = pcaDelay.fit_transform(delay_ics_flat)
            exp_var_delay = np.sum(pcaDelay.explained_variance_ratio_)
            delay_ics = delay_ics.reshape(n_seq, n_trials, 3)

            pcaMove = PCA(n_components=3)
            go_ics = pcaMove.fit_transform(go_ics_flat)
            go_ics = go_ics.reshape(n_seq, n_trials, 3)
            exp_var_go = np.sum(pcaMove.explained_variance_ratio_)

        else:
            exp_var_delay = 1.0
            exp_var_go = 1.0

        # Plot trajectories
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(211, projection="3d")
        # Make colors using jet colormap
        colors = plt.cm.jet(np.linspace(0, 1, n_seq // 2))
        for i in range(n_seq):
            if i < n_seq / 2:
                color = colors[i]
                alpha = 1
                label = target_labels[i]
            else:
                color = colors[i - n_seq // 2]  # Use the same colors as the first half
                alpha = 0.2
                label = None
            ax.scatter(
                *delay_ics[i, :, :].T, alpha=alpha, s=10, color=color, label=label
            )
            # Plot the mean value larger
            ax.scatter(*np.mean(delay_ics[i, :, :], axis=0), color=color, alpha=1, s=30)
        ax.set_title(f"Delay explained variance: {exp_var_delay:.2f}")
        ax.legend()
        plt.tight_layout()

        ax = fig.add_subplot(212, projection="3d")
        for i in range(n_seq):
            if i < n_seq / 2:
                color = colors[i]
                alpha = 1
                label = target_labels[i]
            else:
                color = colors[i - n_seq // 2]  # Use the same colors as the first half
                alpha = 0.2
                label = None
            ax.scatter(*go_ics[i, :, :].T, alpha=alpha, s=10, color=color, label=label)
            # Plot the mean value larger
            ax.scatter(*np.mean(go_ics[i, :, :], axis=0), alpha=1, s=30, color=color)
        ax.set_title(f"Go explained variance: {exp_var_go:.2f}")
        ax.legend()

        plt.tight_layout()

        trainer.loggers[0].experiment.add_figure(
            "delay_go_latents_ICs", fig, global_step=trainer.global_step
        )
        logger.log(
            {
                "delay_go_latent_ICs": wandb.Image(fig),
                "global_step": trainer.global_step,
            }
        )
