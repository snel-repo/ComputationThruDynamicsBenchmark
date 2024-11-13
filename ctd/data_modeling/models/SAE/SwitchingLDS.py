import pytorch_lightning as pl
import torch
from torch import nn

from .loss_func import LossFunc, PoissonLossFunc


class LinearCell(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.linear_ih = nn.Linear(input_size, latent_size)
        self.linear_hh = nn.Linear(latent_size, latent_size)

    def forward(self, input, hidden):
        return self.linear_ih(input) + self.linear_hh(hidden)


class SwitchingLinearCell(nn.Module):
    def __init__(self, input_size, latent_size, num_modes):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_modes = num_modes
        self.cells = nn.ModuleList(
            [LinearCell(input_size, latent_size) for _ in range(num_modes)]
        )
        # A network to compute mode probabilities, taking both input and hidden state
        self.mode_network = nn.Sequential(
            nn.Linear(input_size + latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, input, hidden):
        # Compute mode probabilities
        mode_input = torch.cat(
            [input, hidden], dim=-1
        )  # Shape: (batch_size, input_size + latent_size)
        mode_logits = self.mode_network(mode_input)
        mode_probs = nn.functional.softmax(
            mode_logits, dim=-1
        )  # Shape: (batch_size, num_modes)
        # Compute outputs of each mode
        outputs = []
        for cell in self.cells:
            output = cell(input, hidden)  # Shape: (batch_size, latent_size)
            outputs.append(output.unsqueeze(-1))  # Shape: (batch_size, latent_size, 1)
        outputs = torch.cat(
            outputs, dim=-1
        )  # Shape: (batch_size, latent_size, num_modes)
        # Weight outputs by mode probabilities
        mode_probs = mode_probs.unsqueeze(1)  # Shape: (batch_size, 1, num_modes)
        output = (outputs * mode_probs).sum(dim=-1)  # Shape: (batch_size, latent_size)
        return output, mode_probs


class RNN(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input, h_0):
        hidden = h_0
        states = []
        mode_probs_list = []
        for input_step in input.transpose(0, 1):
            hidden, mode_probs = self.cell(input_step, hidden)
            states.append(hidden)
            mode_probs_list.append(mode_probs)
        states = torch.stack(states, dim=1)  # Shape: (batch_size, seq_len, latent_size)
        mode_probs = torch.stack(
            mode_probs_list, dim=1
        )  # Shape: (batch_size, seq_len, num_modes)
        return states, hidden, mode_probs


class SwitchingLDS(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        encoder_size: int,
        encoder_window: int,
        heldin_size: int,
        heldout_size: int,
        latent_size: int,
        lr: float,
        weight_decay: float,
        dropout: float,
        input_size: int,
        num_modes: int,
        loss_func: LossFunc = PoissonLossFunc(),
    ):
        super().__init__()
        # Instantiate bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=heldin_size,
            hidden_size=encoder_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.readout = nn.Linear(in_features=latent_size, out_features=heldout_size)
        self.ic_linear = nn.Linear(2 * encoder_size, latent_size)
        self.encoder_window = encoder_window
        self.latent_size = latent_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.loss_func = loss_func
        # Use SwitchingLinearCell in the decoder
        self.decoder = RNN(SwitchingLinearCell(input_size, latent_size, num_modes))

    def forward(self, data, inputs):
        # Pass data through the encoder
        _, h_n = self.encoder(data[:, : self.encoder_window, :])
        h_n = torch.cat([*h_n], -1)
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        ic_drop = self.dropout(ic)
        # Pass through the decoder with switching dynamics
        latents, _, mode_probs = self.decoder(inputs, ic_drop)
        B, T, N = latents.shape
        # Map decoder state to data dimension
        rates = self.readout(latents)
        return rates, latents

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.parameters(),
                    "weight_decay": self.weight_decay,
                    "lr": self.lr,
                },
            ],
        )
        return optimizer

    def training_step(self, batch, batch_ix):
        spikes, recon_spikes, inputs, extra, *_ = batch
        # Pass data through the model
        pred_logrates, pred_latents = self.forward(spikes, inputs)
        # Compute the weighted loss
        loss_dict = dict(
            controlled=pred_logrates,
            targets=recon_spikes,
            extra=extra,
        )
        loss = self.loss_func(loss_dict)

        self.log("train/loss_all", loss)

        return loss

    def validation_step(self, batch, batch_ix):
        spikes, recon_spikes, inputs, extra, *_ = batch
        # Pass data through the model
        pred_logrates, latents = self.forward(spikes, inputs)
        loss_dict = dict(
            controlled=pred_logrates,
            targets=recon_spikes,
            extra=extra,
        )

        loss = self.loss_func(loss_dict)

        self.log("valid/loss_all", loss)
        return loss
