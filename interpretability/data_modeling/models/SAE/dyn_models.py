import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from interpretability.data_modeling.models.readouts import FeedForwardNet, FlowReadout


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
        inv_encoder: bool,
        encoder_window: int,
        flow_num_steps: int,
        readout_vf_hidden: int,
        readout_num_layers: int,
        heldin_size: int,
        heldout_size: int,
        latent_size: int,
        lr_readout: float,
        lr_encoder: float,
        lr_decoder: float,
        decay_readout: float,
        decay_encoder: float,
        decay_decoder: float,
        dropout: float,
        readout_type: str,
        input_size: int,
        increment_trial: bool,
        points_per_group: int,
        epochs_per_group: int,
        vf_hidden_size: int,
        vf_num_layers: int,
    ):
        super().__init__()
        # Instantiate bidirectional GRU encoder
        self.inv_encoder = inv_encoder
        self.encoder = nn.GRU(
            input_size=heldin_size,
            hidden_size=encoder_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        # Construct graph based on readout: Linear, INN, or Flow readout
        self.readout_type = readout_type
        # if self.readout_type == "Invert":
        #     self.readout = InvertibleNetNeural(
        #         node_dim=latent_size,
        #         heldin_dim=heldin_size,
        #         heldout_dim=heldout_size,
        #         inn_num_layers=readout_num_layers,
        #     )

        if self.readout_type == "Linear":
            self.readout = nn.Linear(in_features=latent_size, out_features=heldout_size)
        elif self.readout_type == "Flow":
            self.readout = FlowReadout(
                node_dim=latent_size,
                heldin_dim=heldin_size,
                heldout_dim=heldout_size,
                vf_hidden_size=readout_vf_hidden,
                num_layers=readout_num_layers,
                num_steps=flow_num_steps,
            )
        elif self.readout_type == "MLP":
            self.readout = FeedForwardNet(
                input_dim=latent_size,
                output_dim=heldout_size,
                hidden_dim=readout_vf_hidden,
                num_layers=readout_num_layers,
            )
        if inv_encoder:
            self.ic_linear = nn.Linear(2 * encoder_size, heldout_size)
        else:
            self.ic_linear = nn.Linear(2 * encoder_size, latent_size)

        if inv_encoder and self.readout_type == "Linear":
            raise Exception("Linear cant be inverted", "Try another readout")
        if not increment_trial and (points_per_group > 0 or epochs_per_group > 0):
            raise Exception("No incrementing points without a license!", "Try again")
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

    def forward(self, data, inputs):
        # Pass data through the model
        _, h_n = self.encoder(data[:, : self.hparams.encoder_window, :])
        h_n = torch.cat([*h_n], -1)
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        ic_drop = self.dropout(ic)
        if self.inv_encoder:
            ic_drop = self.readout(ic_drop, reverse=True)
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
                    "params": self.readout.parameters(),
                    "weight_decay": self.hparams.decay_readout,
                    "lr": self.hparams.lr_readout,
                },
                {
                    "params": list(self.ic_linear.parameters())
                    + list(self.encoder.parameters()),
                    "weight_decay": self.hparams.decay_encoder,
                    "lr": self.hparams.lr_encoder,
                },
                {
                    "params": self.decoder.parameters(),
                    "weight_decay": self.hparams.decay_decoder,
                    "lr": self.hparams.lr_decoder,
                },
            ],
        )
        return optimizer

    def training_step(self, batch, batch_ix):
        spikes, recon_spikes, inputs, *_ = batch
        # Pass data through the model
        pred_logrates, pred_latents = self.forward(spikes, inputs)
        total_points = pred_logrates.shape[1]

        # Prepare the loss for incremental addition of points
        if self.hparams.increment_trial:
            group_number = int(self.current_epoch / self.hparams.epochs_per_group) + 1
            num_points = min(group_number * self.hparams.points_per_group, total_points)
            self.log("train/num_points", num_points)
            pred_logrates = pred_logrates[:, :num_points, :]
            recon_spikes = recon_spikes[:, :num_points, :]
        # Compute the weighted loss
        loss_nll = F.poisson_nll_loss(pred_logrates, recon_spikes, reduction="none")

        if self.hparams.dataset == "dmfc" or self.hparams.dataset == "malfoy":
            weight1 = torch.ones_like(loss_nll)
            _, _, _, end_idx, *_ = batch
            end_idx = end_idx.detach().cpu().numpy().astype(int)
            # Consider only as many points in loss as trial length
            for i in range(len(end_idx)):
                end1 = np.min([end_idx[i], loss_nll.shape[1]])
                weight1[i, end1:, :] = 0
                weight1[i, :, :] = weight1[i, :, :] / end_idx[i]
            weight1 = weight1.to(self.device)
            loss_nll = loss_nll * weight1

        loss_all_train = torch.mean(loss_nll)
        self.log("train/loss_all_train", loss_all_train)
        if self.readout_type == "LipschitzFlow":
            lip_loss_wt = torch.tensor(1e-6).to(self.device)
            loss_lipschitz = lip_loss_wt * self.readout.lipschitz_loss_term().to(
                self.device
            )
            loss_lipschitz = loss_lipschitz.to(self.device)
            self.log("train/loss_lipschitz", loss_lipschitz)
            loss_all_train += loss_lipschitz

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
            spikes, recon_spikes, inputs, *_ = batch
            # Pass data through the model
            pred_logrates, latents = self.forward(spikes, inputs)

        total_points = pred_logrates.shape[1]
        if self.hparams.increment_trial:
            group_number = int(self.current_epoch / self.hparams.epochs_per_group) + 1
            num_points = min(group_number * self.hparams.points_per_group, total_points)
            self.log("valid/num_points", num_points)
            pred_logrates = pred_logrates[:, :num_points, :]
            recon_spikes = recon_spikes[:, :num_points, :]
        loss_nll = F.poisson_nll_loss(pred_logrates, recon_spikes, reduction="none")

        if self.hparams.dataset == "dmfc" or self.hparams.dataset == "malfoy":
            weight1 = torch.ones_like(loss_nll)
            _, _, _, end_idx, *_ = batch
            end_idx = end_idx.detach().cpu().numpy().astype(int)
            # Consider only as many points in loss as trial length
            for i in range(len(end_idx)):
                end1 = np.min([end_idx[i], loss_nll.shape[1]])
                weight1[i, end1:, :] = 0
                weight1[i, :, :] = weight1[i, :, :] / end_idx[i]
            weight1 = weight1.to(self.device)
            loss_nll = loss_nll * weight1

        loss_all_train = torch.mean(loss_nll)
        self.log("valid/loss_all", loss_all_train)
        if self.readout_type == "LipschitzFlow":
            lip_loss_wt = torch.tensor(1e-6).to(loss_all_train.device)
            loss_lipschitz = lip_loss_wt * self.readout.lipschitz_loss_term().to(
                self.device
            )
            loss_lipschitz = loss_lipschitz.to(loss_all_train.device)
            self.log("valid/loss_lipschitz", loss_lipschitz)
            loss_all_train += loss_lipschitz

        return loss_all_train
