import torch
import torch.nn.functional as F


def _default_2d_array(array):
    return array.reshape(-1, array.shape[-1])


def _default_2d_func(func):
    def wrapper(preds, targets):
        return func(_default_2d_array(preds), _default_2d_array(targets))

    return wrapper


@_default_2d_func
def r2_score(preds, targets):
    target_mean = torch.mean(targets, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    return torch.mean(1 - ss_res / ss_tot)


@_default_2d_func
def linear_regression(preds, targets):
    preds_1 = F.pad(preds, (0, 1), value=1)
    W = preds_1.pinverse() @ targets
    return preds_1 @ W


@_default_2d_func
def regression_r2_score(preds, targets):
    projs = linear_regression(preds, targets)
    return torch.clamp_min(r2_score(projs, targets), -10)


@_default_2d_func
def regression_mse(preds, targets):
    projs = linear_regression(preds, targets)
    return F.mse_loss(projs, targets)


def weighted_loss(loss_fn, preds, targets, weight=1.0):
    loss_all = loss_fn(input=preds, target=targets, reduction="none")
    return torch.mean(weight * loss_all)
