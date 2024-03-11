import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, h_0, num_steps, rev):
        hidden = h_0
        states = []
        vf_out = []
        for input_step in range(num_steps):
            hidden, vf_1 = self.cell(hidden, rev=rev)
            states.append(hidden)
            vf_out.append(vf_1)
        states = torch.stack(states, dim=1)
        vf_out = torch.norm(torch.stack(vf_out, dim=1), dim=2)
        return states, hidden, vf_out


class MLPCell(nn.Module):
    def __init__(self, vf_net):
        super().__init__()
        self.vf_net = vf_net
        self.input_size = 3

    def forward(self, hidden, rev):
        vf_out = 0.1 * self.vf_net(hidden)
        if rev:
            return hidden - vf_out, vf_out
        else:
            return hidden + vf_out, vf_out


class MLPCellScale(nn.Module):
    def __init__(self, vf_net, scale=0.1):
        super().__init__()
        self.vf_net = vf_net
        self.input_size = 3
        self.scale = scale

    def forward(self, hidden, rev):
        vf_out = self.scale * self.vf_net(hidden)
        if rev:
            return hidden - vf_out, vf_out
        else:
            return hidden + vf_out, vf_out


def build_subnet(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 64),
        nn.ReLU(),
        nn.Linear(64, dims_out),
    )


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.network = []
        self.network.append(nn.Linear(input_dim, hidden_dim))
        self.network.append(nn.ReLU())
        for k in range(num_layers - 1):
            self.network.append(nn.Linear(hidden_dim, hidden_dim))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*self.network)

    def forward(self, input):
        return self.network(input)


# class InvertibleNetNeural(nn.Module):
#     def __init__(self, node_dim, heldin_dim, heldout_dim, inn_num_layers):
#         super().__init__()
#         self.node_dim = node_dim
#         self.heldin_dim = heldin_dim
#         self.heldout_dim = heldout_dim
#         self.hidden_dim = max(node_dim, heldout_dim)

#         inn = Ff.SequenceINN(self.hidden_dim)
#         for k in range(inn_num_layers):
#             inn.append(
#                 Fm.AllInOneBlock,
#                 subnet_constructor=build_subnet,
#                 permute_soft=True,
#             )
#         self.network = inn

#     def forward(self, inputs, reverse=False):
#         if not reverse:
#             batch_size, n_steps, n_inputs = inputs.shape
#             assert n_inputs == self.node_dim
#         else:
#             batch_size, n_inputs = inputs.shape
#             assert n_inputs == self.heldout_dim
#         # Pad the inputs if necessary
#         inputs = F.pad(inputs, (0, self.hidden_dim - n_inputs))
#         # Pass the inputs through the network
#         outputs, _ = self.network(inputs.reshape(-1, self.hidden_dim), rev=reverse)
#         if not reverse:
#             outputs = outputs.reshape(batch_size, n_steps, self.hidden_dim)
#             return outputs
#         # Remove padded elements if necessary
#         else:
#             # Trim the final dimension to match the node dimension
#             outputs = outputs[:, : self.node_dim]
#             return outputs


class FlowReadout(nn.Module):
    def __init__(
        self,
        node_dim,
        heldin_dim,
        heldout_dim,
        vf_hidden_size,
        num_layers,
        num_steps,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.heldin_dim = heldin_dim
        self.heldout_dim = heldout_dim

        self.vf_hidden_size = vf_hidden_size
        self.num_layers = num_layers
        self.num_steps = num_steps

        act_func = torch.nn.ReLU
        vector_field = []
        vector_field.append(nn.Linear(self.heldout_dim, self.vf_hidden_size))
        vector_field.append(act_func())
        for k in range(self.num_layers - 1):
            vector_field.append(nn.Linear(vf_hidden_size, vf_hidden_size))
            vector_field.append(act_func())
        vector_field.append(nn.Linear(vf_hidden_size, self.heldout_dim))
        vector_field_net = nn.Sequential(*vector_field)
        self.network = RNN(cell=MLPCell(vf_net=vector_field_net))

    def forward(self, inputs, reverse=False):
        if not reverse:
            batch_size, n_time, n_inputs = inputs.shape
            assert n_inputs == self.node_dim
            inputs = torch.cat(
                [
                    inputs,
                    torch.zeros(
                        batch_size,
                        n_time,
                        self.heldout_dim - self.node_dim,
                        device=inputs.device,
                    ),
                ],
                dim=-1,
            )
        else:
            batch_size, n_inputs = inputs.shape
            assert n_inputs == self.heldout_dim

        # Pass the inputs through the network
        _, outputs, _ = self.network(
            inputs.reshape(-1, self.heldout_dim), num_steps=self.num_steps, rev=reverse
        )
        if not reverse:
            outputs = outputs.reshape(batch_size, n_time, self.heldout_dim)
            return outputs
        else:
            # Trim the final dimension to match the node dimension
            outputs = outputs[:, : self.node_dim]
            return outputs
