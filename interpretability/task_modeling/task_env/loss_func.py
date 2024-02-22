import numpy as np
import torch
import torch.nn as nn


class LossFunc:
    def __init__():
        pass

    def __call__(self, pred, target, act=None):
        pass


class RandomTargetLoss(LossFunc):
    def __init__(
        self, position_loss, pos_weight, act_weight, full_trial_epoch: int = 200
    ):
        self.position_loss = position_loss
        self.action_loss = nn.MSELoss()
        self.pos_weight = pos_weight
        self.act_weight = act_weight
        self.full_trial_epoch = full_trial_epoch

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        epoch = loss_dict["epoch"]
        n_time = pred.shape[1]
        # Gradually increase the percent of the trial to include in the loss
        include_loss = np.ceil(n_time * min(1.0, epoch / self.full_trial_epoch)).astype(
            int
        )
        # inputs = loss_dict["inputs"]
        # TODO: torch.clip
        pos_loss = self.pos_weight * self.position_loss(
            pred[:, :include_loss, :], target[:, :include_loss, :]
        )
        act_loss = self.act_weight * self.action_loss(act, torch.zeros_like(act))
        return pos_loss + act_loss


class MatchTargetLossMSE(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.MSELoss()(pred, target)


class L1LossFunc(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.L1Loss()(pred, target)


class MultiTaskLoss(LossFunc):
    def __init__(self, lat_loss_weight=1e-6):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        latents = loss_dict["latents"]
        # action = loss_dict["actions"]
        inputs = loss_dict["inputs"]
        extras = loss_dict["extra"]
        resp_start = extras[:, 0].long()
        resp_end = extras[:, 1].long()
        recon_loss = nn.MSELoss(reduction="none")(pred, target)
        mask = torch.ones_like(recon_loss)
        mask_lats = torch.ones_like(latents)
        mask[:, 0:5, :] = 0
        for i in range(inputs.shape[0]):
            mask[i, resp_start[i] : resp_end[i], :] = 5.0
            mask[i, resp_start[i] : resp_start[i] + 5, :] = 0.0
            mask[i, resp_end[i] :, :] = 0.0
            mask_lats[i, : resp_end[i], :] = 0

        masked_loss = recon_loss * mask
        lats_loss = (
            nn.MSELoss(reduction="none")(latents, torch.zeros_like(latents)) * mask_lats
        )
        total_loss = masked_loss.sum(dim=1).mean() + lats_loss.sum(dim=1).mean()
        return total_loss
