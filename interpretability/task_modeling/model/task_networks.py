import torch
from torch import nn
from torch.nn import GRUCell


class TaskBase(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def forward(self, inputs, hidden=None):
        n_samples, n_times, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, 1, self.latent_size), device=dev)
        latents = []
        for input_step in range(inputs.shape[1]):
            hidden = self.cell(inputs[:, input_step, :], hidden[:, 0, :])
            latents.append(hidden)
        latents = torch.stack(latents, dim=1)
        output = self.readout(latents)
        return output, latents


class RNN(TaskBase):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__(latent_size, input_size, output_size)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)


class NODE(TaskBase):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
    ):
        super().__init__(
            latent_size=latent_size, input_size=input_size, output_size=output_size
        )
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.cell = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = MLPCell(
            input_size, self.num_layers, self.layer_hidden_size, self.latent_size
        )
        self.readout = nn.Linear(self.latent_size, output_size)


class MLPCell(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + 0.1 * self.vf_net(input_hidden)
