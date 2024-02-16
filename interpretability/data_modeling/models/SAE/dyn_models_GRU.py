import pytorch_lightning as pl
import torch
from torch import nn

from .loss_func import LossFunc, PoissonLossFunc


class RNN(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input, h_0):
        hidden = h_0
        states = []
        for input_step in input.transpose(0, 1):
            hidden = self.cell(input_step, hidden)
            states.append(hidden)
        states = torch.stack(states, dim=1)
        return states, hidden


class GRULatentSAE(pl.LightningModule):
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
        self.save_hyperparameters()
        latent_size = self.hparams.latent_size
        self.loss_func = loss_func
        self.decoder = RNN(nn.GRUCell(input_size, latent_size))

    def forward(self, data, inputs):
        # Pass data through the model
        _, h_n = self.encoder(data[:, : self.hparams.encoder_window, :])
        h_n = torch.cat([*h_n], -1)
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        ic_drop = self.dropout(ic)
        # Evaluate the NeuralODE
        latents, _ = self.decoder(inputs, ic_drop)
        B, T, N = latents.shape
        # Map decoder state to data dimension
        rates = self.readout(latents)
        return rates, latents

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.lr,
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
