import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchdyn.core import NeuralODE


class RNN(nn.Module):
    def __init__(self, cell, latent_size):
        super().__init__()
        self.cell = cell
        self.latent_size = latent_size

    def forward(self, input):
        n_samples, n_times, n_inputs = input.shape
        dev = input.device
        hidden = torch.zeros((n_samples, self.latent_size), device= dev)
        states = []
        for input_step in input.transpose(0, 1):
            hidden = self.cell(input_step, hidden)
            states.append(hidden)
        states = torch.stack(states, dim=1)
        return states, hidden



class MLPCell(nn.Module):
    def __init__(self, vf_net, state_2_state):
        super().__init__()
        self.vf_net = vf_net
        self.state_2_state = state_2_state

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        if self.state_2_state:
            return self.vf_net(input_hidden)
        else:
            return hidden + 0.1*self.vf_net(input_hidden)

class TaskTrainedNODE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        output_size: int,
        learning_rate: float,
        weight_decay: float,
        vf_hidden_size: int,
        vf_depth: int,
        state_2_state: bool, 
    ):
        super().__init__()

        act_func = torch.nn.ReLU
        vector_field = []
        vector_field.append(nn.Linear(latent_size +input_size, vf_hidden_size))
        vector_field.append(act_func())
        for k in range(vf_depth - 1):
            vector_field.append(nn.Linear(vf_hidden_size, vf_hidden_size))
            vector_field.append(act_func())
        vector_field.append(nn.Linear(vf_hidden_size, latent_size))
        vector_field_net = nn.Sequential(*vector_field)
        self.model = RNN(MLPCell(vector_field_net, state_2_state), latent_size= latent_size)

        self.readout = nn.Linear(latent_size, output_size)

        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.vf_hidden_size = vf_hidden_size
        self.vf_depth = vf_depth
        self.state_2_state = state_2_state
        
        self.save_hyperparameters()

    def forward(self, inputs):
        # Pass data through the model
        latents, _ = self.model(inputs)
        output = self.readout(latents)
        return output, latents

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_ix):
        states, inputs, comb, train_inds = batch
        # Pass data through the model
        pred_states, _ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = F.mse_loss(pred_states, states)
        self.log("train/loss", loss_all)

        return loss_all

    def validation_step(self, batch, batch_ix):
        states, inputs, comb, valid_inds = batch
        # Pass data through the model
        pred_states,_ = self.forward(inputs)
        # Compute the weighted loss
        loss_all = F.mse_loss(pred_states, states)
        self.log("valid/loss", loss_all)
        return loss_all