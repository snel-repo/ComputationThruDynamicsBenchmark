import pytorch_lightning as pl
import torch
from gymnasium import Env
from torch import nn

from interpretability.task_modeling.model.modules.loss_func import LossFunc


class TaskTrainedWrapper(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        input_size=None,
        output_size=None,
        task_env: Env = None,
        model: nn.Module = None,
        state_label: str = None,
        loss_func: LossFunc = None,
    ):
        super().__init__()
        self.task_env = task_env
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.state_label = state_label
        self.loss_func = loss_func
        self.save_hyperparameters()

    def set_environment(self, task_env: Env):
        self.task_env = task_env
        self.input_size = (
            task_env.goal_space.shape[0] + task_env.observation_space.shape[0]
        )
        self.output_size = task_env.action_space.shape[0]

    def set_model(self, model: nn.Module):
        self.model = model
        self.latent_size = model.latent_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(self, ics, inputs, targets=None):

        # Pass data through the model
        batch_size = ics.shape[0]

        # If we are in a coupled environment, set the environment state
        if self.task_env.coupled_env:
            env_states, info = self.task_env.reset(
                batch_size=batch_size, ic_state=ics, target_state=targets[:, 0, :]
            )
            env_state_list = []
        else:
            env_states = None
            env_state_list = None

        # Set the model's initial hidden state
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(batch_size=batch_size).to(self.device)
        else:
            hidden = torch.zeros(batch_size, self.latent_size).to(self.device)

        # Latents are the hidden states of the model
        latents = []
        controlled = []
        actions = []

        count = 0
        terminated = False
        while not terminated and len(controlled) < self.task_env.n_timesteps:
            # Get the appropriate model input for coupled and decoupled envs
            if self.task_env.coupled_env:
                model_input = torch.hstack((env_states, inputs[:, count, :]))
            else:
                model_input = inputs[:, count, :]
            # Run the model on the input
            action, hidden = self.model(model_input, hidden)
            # If we are in a coupled environment, step the environment
            if self.task_env.coupled_env:
                self.task_env.set_goal(targets[:, count, :])
                env_states, _, terminated, _, info = self.task_env.step(
                    action=action, inputs=inputs[:, count, :]
                )
                controlled.append(info["states"][self.state_label])
                actions.append(action)
                env_state_list.append(env_states)
            # If we are in a decoupled environment, just record the action
            else:
                controlled.append(action)
                actions.append(action)
            latents.append(hidden)
            count += 1
        controlled = torch.stack(controlled, dim=1)
        latents = torch.stack(latents, dim=1)
        actions = torch.stack(actions, dim=1)
        if self.task_env.coupled_env:
            states = torch.stack(env_state_list, dim=1)
        else:
            states = None

        output_dict = {
            "controlled": controlled,  # BETTER VARIABLE NAME?
            "latents": latents,
            "actions": actions,
            "states": states,
        }
        return output_dict

    def training_step(self, batch, batch_ix):
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]

        # Pass data through the model
        output_dict = self.forward(ics, inputs, targets)
        # Compute the weighted loss
        loss_dict = {
            "controlled": output_dict["controlled"],
            "actions": output_dict["actions"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
        }
        loss_all = self.loss_func(loss_dict)
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        ics = batch[0]
        inputs = batch[1]
        targets = batch[2]
        conds = batch[4]
        extras = batch[5]

        # Pass data through the model
        output_dict = self.forward(ics, inputs, targets)
        # Compute the weighted loss
        loss_dict = {
            "controlled": output_dict["controlled"],
            "actions": output_dict["actions"],
            "targets": targets,
            "inputs": inputs,
            "conds": conds,
            "extra": extras,
            # TODO: Pass in logger to log seperate losses
            # or make loss_func return seperate losses as a dict?
        }
        loss_all = self.loss_func(loss_dict)
        self.log("valid/loss", loss_all)

        return loss_all
