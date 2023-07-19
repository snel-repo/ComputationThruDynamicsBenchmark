import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn



class TaskTrainedDecoupled(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float,
        weight_decay: float,
        model: nn.Module = None,
        readout: nn.Module = None,
        loss_func = None,
    ):
        super().__init__()
 
        self.model = model
        self.readout = readout

        self.input_size = input_size
        self.output_size = output_size
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
    
    def set_model(self, model):
        self.model = model
        self.readout = nn.Linear(model.latent_size, self.output_size)

    def forward(self, inputs):
        # Pass data through the model
        latents, _ = self.model(inputs)
        output = self.readout(latents)
        return output, latents
    
    def training_step(self, batch, batch_ix):
        states, inputs, comb, train_inds = batch
        # Pass data through the model
        pred_states, _ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = self.loss_func(pred_states, states)
        self.log("train/loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_ix):
        states, inputs, comb, valid_inds = batch
        # Pass data through the model
        pred_states,_ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = self.loss_func(pred_states, states)
        self.log("valid/loss", loss_all)
        return loss_all