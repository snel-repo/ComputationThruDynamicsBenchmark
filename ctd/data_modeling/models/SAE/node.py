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


class MLPCell(nn.Module):
    def __init__(self, vf_net, input_size):
        super().__init__()
        self.vf_net = vf_net
        self.input_size = input_size

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        vf_out = 0.1 * self.vf_net(input_hidden)
        return hidden + vf_out


class NODELatentSAE(pl.LightningModule):
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
        vf_hidden_size: int,
        vf_num_layers: int,
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
        self.ic_linear = nn.Linear(2 * encoder_size, latent_size)
        self.save_hyperparameters()

        act_func = torch.nn.ReLU
        latent_size = self.hparams.latent_size
        vector_field = []
        vector_field.append(nn.Linear(latent_size + input_size, vf_hidden_size))
        vector_field.append(act_func())
        for k in range(self.hparams.vf_num_layers - 1):
            vector_field.append(nn.Linear(vf_hidden_size, vf_hidden_size))
            vector_field.append(act_func())
        vector_field.append(nn.Linear(vf_hidden_size, latent_size))
        vector_field_net = nn.Sequential(*vector_field)
        self.decoder = RNN(MLPCell(vector_field_net, input_size))
        self.readout = nn.Linear(in_features=latent_size, out_features=heldout_size)
        self.loss_func = loss_func
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, data, inputs):
        # Pass data through the model
        _, h_n = self.encoder(data[:, : self.hparams.encoder_window, :])
        h_n = torch.cat([*h_n], -1)
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        # Evaluate the NeuralODE
        latents, _ = self.decoder(inputs, ic)
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
        loss_all_train = self.loss_func(loss_dict)
        self.log("train/loss_all_train", loss_all_train)

        return loss_all_train

    def validation_step(self, batch, batch_ix):
        if len(batch) == 1:
            (spikes,) = batch
            # Pass data through the model
            pred_logrates, latents = self.forward(spikes)
            # Isolate heldin predictions
            _, n_obs, n_heldin = spikes.shape
            pred_logrates = pred_logrates[:, :n_obs, :n_heldin]
            recon_spikes = spikes
        else:
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
