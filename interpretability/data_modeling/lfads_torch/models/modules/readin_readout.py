import math

import torch
from torch import nn

# import h5py


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, out_features):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


class Flow(nn.Module):
    def __init__(
        self,
        readout_num_layers,
        readout_hidden_size,
        in_features,
        out_features,
        flow_num_steps,
    ):
        super(Flow, self).__init__()

        self.flow_num_steps = flow_num_steps
        self.neural_dim = out_features
        self.latent_dim = in_features
        self.network = MLP(
            in_features=self.neural_dim,
            hidden_size=readout_hidden_size,
            num_layers=readout_num_layers,
            out_features=self.neural_dim,
        )

    def forward(self, Z, reverse=False):
        if reverse:
            B, N = Z.shape
            # Trim the last elements to make it D dimensional
            for _ in range(self.flow_num_steps):
                deltaZ = self.network(Z)
                Z = Z - deltaZ
            return Z[:, : self.latent_dim]
        else:
            B, T, D = Z.shape
            D = self.latent_dim
            N = self.neural_dim
            # Padding the tensor
            Z = nn.functional.pad(Z, (0, N - D))
            # Reshape the tensor to be a BT x N tensor
            Z = Z.reshape(-1, N)
            # Applying the MLP for flow_num_steps times
            for _ in range(self.flow_num_steps):
                deltaZ = self.network(Z)
                Z = Z + deltaZ
            # Reshape the tensor back to BxTxN
            Z = Z.reshape(B, T, self.neural_dim)
            return Z


# class PCRInitModuleList(nn.ModuleList):
#     def __init__(self, inits_path: str, modules: list[nn.Module]):
#         super().__init__(modules)
#         # Pull pre-computed initialization from the file, assuming correct order
#         with h5py.File(inits_path, "r") as h5file:
#             weights = [v["/" + k + "/matrix"][()] for k, v in h5file.items()]
#             biases = [v["/" + k + "/bias"][()] for k, v in h5file.items()]
#         # Load the state dict for each layer
#         for layer, weight, bias in zip(self, weights, biases):
#             state_dict = {"weight": torch.tensor(weight), "bias": torch.tensor(bias)}
#             layer.load_state_dict(state_dict)
