import torch
from torch import nn


class NODEPolicy(nn.Module):
    def __init__(
        self,
        latent_size: int,
        num_layers: int,
        layer_hidden_size: int,
        input_size=None,
        output_size=None,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.input_size = input_size
        self.output_size = output_size

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        num_layers = self.num_layers
        latent_size = self.latent_size
        layer_hidden_size = self.layer_hidden_size

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
        input_hidden = torch.cat([h0, x], dim=1)
        h = h0 + 0.1 * self.vf_net(input_hidden)
        u = self.sigmoid(self.fc(h))
        return u, h


class RNNPolicy(nn.Module):
    def __init__(self, latent_size: int, input_size=None, output_size=None):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = 1

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.gru = torch.nn.GRU(input_size, self.latent_size, 1, batch_first=True)
        self.fc = torch.nn.Linear(self.latent_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                torch.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                torch.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                torch.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                torch.nn.init.zeros_(param)
            elif name == "fc.weight":
                torch.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                torch.nn.init.constant_(param, -5.0)
            else:
                raise ValueError

    def forward(self, x, h0):
        y, latents = self.gru(x[:, None, :], h0[None, :, :])
        output = self.sigmoid(self.fc(y)).squeeze(dim=1)
        latents = latents.squeeze(dim=0)
        return output, latents

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(batch_size, self.latent_size).zero_()
        return hidden


class VanillaRNNPolicy(nn.Module):
    def __init__(self, latent_size: int, input_size=None, output_size=None):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = 1

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.gru = torch.nn.RNN(input_size, self.latent_size, 1, batch_first=True)
        self.fc = torch.nn.Linear(self.latent_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                torch.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                torch.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                torch.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                torch.nn.init.zeros_(param)
            elif name == "fc.weight":
                torch.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                torch.nn.init.constant_(param, -5.0)
            else:
                raise ValueError

    def forward(self, x, h0):
        y, latents = self.gru(x[:, None, :], h0[None, :, :])
        output = self.sigmoid(self.fc(y)).squeeze(dim=1)
        latents = latents.squeeze(dim=0)
        return output, latents

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(batch_size, self.latent_size).zero_()
        return hidden
