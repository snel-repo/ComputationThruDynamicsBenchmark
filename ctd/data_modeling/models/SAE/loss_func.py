import torch
import torch.nn.functional as F


class LossFunc:
    def __init__():
        pass

    def __call__(self, pred, target):
        pass


class PoissonLossFunc(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return F.poisson_nll_loss(pred, target)


class MultiTaskPoissonLossFunc(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        extras = loss_dict["extra"]
        end_ind = extras[:, 1]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        loss_all = F.poisson_nll_loss(pred, target, reduction="none")
        weights = torch.ones_like(loss_all)
        for i in range(len(end_ind)):
            weights[i, int(end_ind[i]) :, :] = 0
        # Normalize each trial by the number of time steps
        weights = weights / weights.sum(dim=1, keepdim=True)
        loss = torch.mean(loss_all * weights)
        return loss
