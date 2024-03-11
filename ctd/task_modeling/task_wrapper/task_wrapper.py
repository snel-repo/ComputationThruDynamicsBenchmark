import pytorch_lightning as pl
import torch
from gymnasium import Env
from torch import nn


class TaskTrainedWrapper(pl.LightningModule):
    """Wrapper for a task trained model

    Handles the training and validation steps for a task trained model.

    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        input_size=None,
        output_size=None,
        task_env: Env = None,
        model: nn.Module = None,
    ):
        """Initialize the wrapper

        args:
            learning_rate (float):
                The learning rate for the optimizer
            weight_decay (float):
                The weight decay for the optimizer
            input_size (int):
                The size of the input to the model (i.e., 3 for 3BFF)
            output_size (int):
                The size of the output of the model (i.e., what the model is predicting)
            task_env (Env):
                The environment to simulate
            model (nn.Module):
                The model to train
            loss_func (LossFunc):
                The loss function to use
                - see loss_func.py for examples

        """
        super().__init__()
        self.task_env = task_env
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def set_environment(self, task_env: Env):
        """Set the environment for the training pipeline"""

        self.task_env = task_env
        self.input_size = (
            task_env.context_inputs.shape[0] + task_env.observation_space.shape[0]
        )
        self.output_size = task_env.action_space.shape[0]
        self.loss_func = task_env.loss_func
        if hasattr(task_env, "state_label"):
            self.state_label = task_env.state_label

    def set_model(self, model: nn.Module):
        """Set the model for the training pipeline"""
        self.model = model
        self.latent_size = model.latent_size

    def configure_optimizers(self):
        """Configure the optimizer"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(self, ics, inputs, inputs_to_env=None):
        """Pass data through the model
        args:
            ics (torch.Tensor):
                The initial conditions for the environment
            inputs (torch.Tensor):
                The inputs to the model
            targets (torch.Tensor):
                The targets for the model
                (i.e., the outputs of the environment or MotorNet goal)

        """
        batch_size = ics.shape[0]

        # If a coupled environment, set the environment state
        if self.task_env.coupled_env:
            options = {"ic_state": ics}
            env_states, info = self.task_env.reset(
                batch_size=batch_size, options=options
            )
            env_state_list = []
            joints = []  # Joint angles
        else:
            env_states = None
            env_state_list = None

        # Call initializations (if they exist)
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(batch_size=batch_size).to(self.device)
        else:
            hidden = torch.zeros(batch_size, self.latent_size).to(self.device)

        latents = []  # Latent activity of TT model
        controlled = []  # Variable controlled by model
        actions = []  # Actions taken by model (sometimes = controlled)

        count = 0
        terminated = False
        while not terminated and len(controlled) < self.task_env.n_timesteps:

            # Build inputs to model
            if self.task_env.coupled_env:
                model_input = torch.hstack((env_states, inputs[:, count, :]))
            else:
                model_input = inputs[:, count, :]

            # Produce an action and a hidden state
            action, hidden = self.model(model_input, hidden)

            # Apply action to environment (for coupled)
            if self.task_env.coupled_env:
                # Apply external loads (if they exist)
                if inputs_to_env is not None:
                    env_states, _, terminated, _, info = self.task_env.step(
                        action=action,
                        inputs=inputs[:, count, :],
                        endpoint_load=inputs_to_env[:, count, :],
                    )
                else:
                    env_states, _, terminated, _, info = self.task_env.step(
                        action=action, inputs=inputs[:, count, :]
                    )
                controlled.append(info["states"][self.state_label])
                joints.append(info["states"]["joint"])
                actions.append(action)
                env_state_list.append(env_states)
            else:
                controlled.append(action)
                actions.append(action)

            latents.append(hidden)
            count += 1

        # Compile outputs
        controlled = torch.stack(controlled, dim=1)
        latents = torch.stack(latents, dim=1)
        actions = torch.stack(actions, dim=1)
        if self.task_env.coupled_env:
            states = torch.stack(env_state_list, dim=1)
            joints = torch.stack(joints, dim=1)
        else:
            states = None
            joints = None

        output_dict = {
            "controlled": controlled,  # BETTER VARIABLE NAME?
            "latents": latents,
            "actions": actions,
            "states": states,
            "joints": joints,
        }
        return output_dict

    def training_step(self, batch, batch_ix):
        # Get the batch data
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]
        inputs_to_env = batch[6]

        # Pass data through the model
        output_dict = self.forward(ics, inputs, inputs_to_env)

        # Compute the weighted loss
        loss_dict = {
            "controlled": output_dict["controlled"],
            "latents": output_dict["latents"],
            "actions": output_dict["actions"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            "epoch": self.current_epoch,
        }

        # Compute the loss using the loss function object
        loss_all = self.loss_func(loss_dict)
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]
        inputs_to_env = batch[6]

        # Pass data through the model
        output_dict = self.forward(ics, inputs, inputs_to_env=inputs_to_env)

        # Compute the weighted loss
        loss_dict = {
            "controlled": output_dict["controlled"],
            "actions": output_dict["actions"],
            "latents": output_dict["latents"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            "epoch": self.current_epoch,
        }

        # Compute the loss using the loss function object
        loss_all = self.loss_func(loss_dict)
        self.log("valid/loss", loss_all)
        return loss_all
