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


class RNN(nn.Module):
    def __init__(self, cells):
        super().__init__()
        self.cells = nn.ModuleList(cells)
        self.num_LDS = len(cells)

    def forward(self, input, h_0, selection_probs):
        hidden = h_0
        states = []
        for input_step in input.transpose(0, 1):
            weighted_hidden = torch.zeros_like(hidden)
            for i in range(self.num_LDS):
                weighted_hidden += selection_probs[:, i : i + 1] * self.cells[i](
                    input_step, hidden
                )
            hidden = weighted_hidden
            states.append(hidden)
        states = torch.stack(states, dim=1)
        return states, hidden


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
        num_LDS: int,
        loss_func: LossFunc = PoissonLossFunc(),
    ):
        super().__init__()
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

        self.num_LDS = num_LDS
        self.selection_mlp = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_LDS),
            nn.Softmax(dim=-1),
        )

        cells = [LinearCell(input_size, latent_size) for _ in range(num_LDS)]
        self.decoder = RNN(cells)
        self.loss_func = loss_func

    def forward(self, data, inputs):
        device = data.device
        self.decoder.to(
            device
        )  # Ensure the decoder (and its cells) are moved to the same device

        data, inputs = data.to(device), inputs.to(device)

        _, h_n = self.encoder(data[:, : self.encoder_window, :])
        h_n = torch.cat([*h_n], -1)
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        ic_drop = self.dropout(ic)

        selection_probs = self.selection_mlp(ic_drop)
        latents, _ = self.decoder(inputs, ic_drop, selection_probs)

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
        spikes, recon_spikes, inputs, extra = (
            spikes.to(self.device),
            recon_spikes.to(self.device),
            inputs.to(self.device),
            extra.to(self.device),
        )
        pred_logrates, pred_latents = self.forward(spikes, inputs)
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
        spikes, recon_spikes, inputs, extra = (
            spikes.to(self.device),
            recon_spikes.to(self.device),
            inputs.to(self.device),
            extra.to(self.device),
        )
        pred_logrates, latents = self.forward(spikes, inputs)
        loss_dict = dict(
            controlled=pred_logrates,
            targets=recon_spikes,
            extra=extra,
        )

        loss = self.loss_func(loss_dict)

        self.log("valid/loss_all", loss)
        return loss
