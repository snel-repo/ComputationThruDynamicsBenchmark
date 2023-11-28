import torch
import torch.nn as nn


class LossFunc:
    def __init__():
        pass

    def __call__(self, pred, target, act=None):
        pass


class RandomTargetLoss(LossFunc):
    def __init__(self, position_loss, pos_weight, act_weight):
        self.position_loss = position_loss
        self.pos_weight = pos_weight
        self.act_weight = act_weight

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        pos_loss = self.pos_weight * self.position_loss(pred, target)
        act_loss = self.act_weight * nn.MSELoss()(act, torch.zeros_like(act))
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
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        inputs = loss_dict["inputs"]
        recon_loss = nn.MSELoss(reduction="none")(pred, target)
        mask = (inputs.sum(dim=2, keepdim=True) != 0).float()
        masked_loss = recon_loss * mask
        total_loss = masked_loss.sum(dim=1).mean()
        return total_loss
