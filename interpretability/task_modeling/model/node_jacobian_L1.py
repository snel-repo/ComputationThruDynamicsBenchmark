import torch
from torch import nn


def compute_generator_jacobian_image_optimized(
    model, embedding, epsilon_scale=0.001, device="cpu"
):
    raw_jacobian = compute_generator_jacobian_optimized(
        model, embedding, epsilon_scale, device
    )
    # shape is (latent_size, batch_size, numchannels = 1, im_size, im_size)
    jacobian = torch.sum(raw_jacobian, dim=2, keepdim=True)
    return jacobian


# output shape is (latent_dim, batch_size, model_output_shape)
def compute_generator_jacobian_optimized(
    model, embedding, epsilon_scale=0.001, device="cpu"
):
    batch_size = embedding.shape[0]
    latent_dim = embedding.shape[1]
    # repeat "tiles" like ABCABCABC (not AAABBBCCC)
    # note that we detach the embedding here, so we should hopefully
    # not be pulling our gradients further back than we intend
    encoding_rep = embedding.repeat(latent_dim + 1, 1).detach().clone()
    # define our own repeat to work like "AAABBBCCC"
    delta = (
        torch.eye(latent_dim)
        .reshape(latent_dim, 1, latent_dim)
        .repeat(1, batch_size, 1)
        .reshape(latent_dim * batch_size, latent_dim)
    )
    delta = torch.cat((delta, torch.zeros(batch_size, latent_dim))).to(device)
    epsilon = epsilon_scale
    encoding_rep += epsilon * delta
    recons = model(encoding_rep)
    temp_calc_shape = [latent_dim + 1, batch_size] + list(recons.shape[1:])
    recons = recons.reshape(temp_calc_shape)
    recons = (recons[:-1] - recons[-1]) / epsilon
    return recons


def jacobian_loss_function(model, mu, device):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(
        model, mu, epsilon_scale=0.001, device=device
    )
    # print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    loss = torch.sum(torch.abs(jacobian)) / batch_size
    assert len(loss.shape) == 0, "loss should be a scalar"
    return loss


# TODO: Rename lowercase
class NODE(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
        input_lat_size=4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.input_lat_size = input_lat_size
        self.generator = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.generator = MLPCell(
            self.input_lat_size,
            self.num_layers,
            self.layer_hidden_size,
            self.latent_size,
        )
        self.readin = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_lat_size),
        )
        # Make readout simple MLP
        self.readout = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size),
        )

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros(
                (n_samples, self.latent_size), device=dev, requires_grad=True
            )
        inputs = self.readin(inputs)
        hidden, jacL1 = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden, jacL1


class MLPCell(nn.Module):
    def __init__(self, input_lat_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.input_size = input_lat_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    nn.Linear(self.input_size + latent_size, layer_hidden_size)
                )
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, inputs, hidden):
        inputs_hidden = torch.cat([hidden, inputs], dim=1)
        vf_out = self.vf_net(inputs_hidden)
        dev = vf_out.device
        jacL1 = jacobian_loss_function(self.vf_net, inputs_hidden, dev)
        hidden = hidden + 0.1 * vf_out
        return hidden, jacL1
