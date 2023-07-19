import torch
from torch import nn

class BiasedNODECell(torch.nn.Module):
    def __init__(self, input_size: int, latent_size: int, output_size: int, num_layers: int, layer_hidden_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
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
        self.fc = torch.nn.Linear(latent_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x, h0):
        input_hidden = torch.cat([x, h0], dim=1)
        h = h0 + 0.1*self.vf_net(input_hidden)
        u = self.sigmoid(self.fc(h)).squeeze(dim=1)
        return u, h
