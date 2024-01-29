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
        self.action_loss = nn.MSELoss()
        self.pos_weight = pos_weight
        self.act_weight = act_weight

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        # TODO: torch.clip
        pos_loss = self.pos_weight * self.position_loss(pred, target)
        act_loss = self.act_weight * self.action_loss(act, torch.zeros_like(act))
        return pos_loss + act_loss


class RandomTargetLossJacL1(LossFunc):
    def __init__(self, position_loss, pos_weight, act_weight, jac_weight):
        self.position_loss = position_loss
        self.action_loss = nn.MSELoss()
        self.pos_weight = pos_weight
        self.act_weight = act_weight
        self.jac_weight = jac_weight

    def __call__(self, loss_dict):
        phase = loss_dict["phase"]
        logger = loss_dict["logger"]
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        jac = loss_dict["jacL1"]
        # inputs = loss_dict["inputs"]
        # TODO: torch.clip
        pos_loss = self.pos_weight * self.position_loss(pred, target)
        act_loss = self.act_weight * self.action_loss(act, torch.zeros_like(act))
        jac_loss = self.jac_weight * jac
        loss_dict = {
            f"{phase}/pos_loss": pos_loss,
            f"{phase}/act_loss": act_loss,
            f"{phase}/jac_loss": jac_loss,
        }
        logger.log_metrics(loss_dict)
        return pos_loss + act_loss + jac_loss


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


def add_c_mask(self, pre_offs, post_ons):
    """Add a cost mask.
    Usually there are two periods, pre and post response
    Scale the mask weight for the post period so in total it's as important
    as the pre period
    """
    pre_on = int(100 / self.dt)  # never check the first 100ms
    pre_offs = self.expand(
        pre_offs
    )  # Assuming self.expand is compatible with PyTorch tensors
    post_ons = self.expand(post_ons)

    c_mask = torch.zeros(
        (self.tdim, self.batch_size, self.n_output), dtype=self.float_type
    )
    for i in range(self.batch_size):
        c_mask[post_ons[i] :, i, :] = 5.0
        c_mask[pre_on : pre_offs[i], i, :] = 1.0
    c_mask[:, :, 0] *= 2.0  # Fixation is important
    self.c_mask = c_mask.view((self.tdim * self.batch_size, self.n_output))


class MultiTaskLoss(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        inputs = loss_dict["inputs"]
        extras = loss_dict["extra"]
        resp_start = extras[:, 0].long()
        resp_end = extras[:, 1].long()
        recon_loss = nn.MSELoss(reduction="none")(pred, target)
        mask = torch.ones_like(recon_loss)
        mask[:, 0:5, :] = 0
        for i in range(inputs.shape[0]):
            mask[i, resp_start[i] : resp_end[i], :] = 5.0
            mask[i, resp_start[i] : resp_start[i] + 5, :] = 0.0
            mask[i, resp_end[i] :, :] = 0.0

        masked_loss = recon_loss * mask
        total_loss = masked_loss.sum(dim=1).mean()
        return total_loss
