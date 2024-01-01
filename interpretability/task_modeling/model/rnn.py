import torch
from torch import nn
from torch.nn import GRUCell, RNNCell


class GRU_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden):
        n_samples, n_inputs = inputs.shape
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class Vanilla_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, 1, self.latent_size), device=dev)
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden
