import torch
from torch import nn
from torch.nn import GRUCell, RNNCell

"""
All models must meet a few requirements
    1. They must have an init_model method that takes
    input_size and output_size as arguments
    2. They must have a forward method that takes inputs and hidden
    as arguments and returns output and hidden for one time step
    3. They must have a cell attribute that is the recurrent cell
    4. They must have a readout attribute that is the output layer
    (mapping from latent to output)
"""


class GRU_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class NoisyGRU_RNN(nn.Module):
    def __init__(
        self, latent_size, input_size=None, output_size=None, noise_level=0.05
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class DriscollRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = (1 - self.gamma) * self.recW(hidden) + self.gamma * self.act_func(
            self.recW(hidden) + self.inpW(inputs) + self.bias + noise
        )
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
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden
