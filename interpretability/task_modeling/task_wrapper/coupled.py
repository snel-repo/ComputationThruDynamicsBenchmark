import pytorch_lightning as pl
import torch
from gymnasium.envs import Environment
from torch import nn

from interpretability.task_modeling.model.modules.loss_func import LossFunc


class TaskTrainedCoupled(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float,
        weight_decay: float,
        latent_size: int = None,
        task_env: Environment = None,
        model: nn.Module = None,
        state_label: str = None,
        loss_func: LossFunc = None,
    ):
        super().__init__()
        self.task_env = task_env
        self.model = model
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.state_label = state_label
        self.loss_func = loss_func
        self.save_hyperparameters()

    def set_task_env(self, task_env: Environment):
        self.task_env = task_env

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

    def forward(self, joints, goal):
        # TODO: Make coupled loop more abstract for non-MotorNet tasks
        terminated = False
        # Pass data through the model
        batch_size = joints.shape[0]
        obs, info = self.task_env.reset(
            batch_size=batch_size, joint_state=joints, goal=goal
        )
        # if the model has a init_hidden method, call it
        if hasattr(self.model, "init_hidden"):
            h = self.model.init_hidden(batch_size=batch_size).to(self.device)
        else:
            h = torch.zeros(batch_size, self.latent_size).to(self.device)
        h_all = [h]
        xy = [info["states"][self.state_label][:, None, :]]
        tg = [info["goal"][:, None, :]]
        actions = []
        while not terminated:
            action, h = self.model(obs, h)  # TODO: Pop out action from model
            obs, reward, terminated, truncated, info = self.task_env.step(action=action)
            xy.append(info["states"][self.state_label][:, None, :])
            tg.append(info["goal"][:, None, :])
            actions.append(action)
            h_all.append(h)
        xy = torch.cat(xy, dim=1)
        tg = torch.cat(tg, dim=1)
        actions = torch.stack(actions, dim=1)
        h_all = torch.stack(h_all, dim=1)
        return xy, tg, h_all, actions

    def training_step(self, batch, batch_ix):
        joints = batch[0]
        goal = batch[1]
        # Pass data through the model
        xy, tg, latents, actions = self.forward(joints, goal)
        # Compute the weighted loss
        loss_all = self.loss_func(xy, tg, actions)
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        joints = batch[0]
        goal = batch[1]
        # Pass data through the model
        xy, tg, latents, actions = self.forward(joints, goal)
        # Compute the weighted loss
        loss_all = self.loss_func(xy, tg, actions)
        self.log("valid/loss", loss_all)
        return loss_all
