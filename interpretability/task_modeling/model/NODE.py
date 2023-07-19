import torch
from torch import nn

class NODE(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.cell = MLPCell(input_size, num_layers, layer_hidden_size, latent_size)
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
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
        self.relu = nn.ReLU()
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + 0.1*self.vf_net(input_hidden)