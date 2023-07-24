import pytorch_lightning as pl
import torch
from torch import nn


class TaskTrainedDecoupled(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        model: nn.Module = None,
        readout: nn.Module = None,
        loss_func=None,
    ):
        super().__init__()

        self.model = model
        self.readout = readout

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def set_environment(self, task_env):
        pass

    def set_model(self, model):
        self.model = model

    def forward(self, inputs):
        # TODO: Move looping logic out of model,
        # TODO: Check to see if basic torch models will work here
        # Pass data through the model
        n_samples, n_times, n_inputs = inputs.shape
        dev = inputs.device
        hidden = torch.zeros((n_samples, 1, self.model.latent_size), device=dev)
        latents = []
        outputs = []
        for i in range(n_times):
            output, hidden = self.model(inputs[:, i : i + 1, :], hidden)
            latents.append(hidden)
            outputs.append(output)
        latents = torch.hstack(latents)
        outputs = torch.hstack(outputs)
        return outputs, latents

    def training_step(self, batch, batch_ix):
        inputs, outputs, train_inds = batch
        # Pass data through the model
        pred_outputs, _ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = self.loss_func(pred_outputs, outputs)
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        outputs, inputs, valid_inds = batch
        # Pass data through the model
        pred_outputs, _ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = self.loss_func(pred_outputs, outputs)
        self.log("valid/loss", loss_all)
        return loss_all
