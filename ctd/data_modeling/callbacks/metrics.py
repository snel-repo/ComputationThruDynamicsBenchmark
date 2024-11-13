import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.nn.functional import poisson_nll_loss


def _default_2d_array(array):
    return array.reshape(-1, array.shape[-1])


def _default_2d_func(func):
    def wrapper(preds, targets):
        return func(_default_2d_array(preds), _default_2d_array(targets))

    return wrapper


# @_default_2d_func
# def r2_score(preds, targets):
#     target_mean = torch.mean(targets, dim=0)
#     ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
#     ss_res = torch.sum((targets - preds) ** 2, dim=0)
#     return torch.mean(1 - ss_res / ss_tot)


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


def bits_per_spike(preds, targets):
    """
    Computes BPS for n_samples x n_timesteps x n_neurons arrays.
    Preds are logrates and targets are binned spike counts.
    """
    nll_model = poisson_nll_loss(preds, targets, full=True, reduction="sum")
    nll_null = poisson_nll_loss(
        torch.mean(targets, dim=(0, 1), keepdim=True),
        targets,
        log_input=False,
        full=True,
        reduction="sum",
    )
    return (nll_null - nll_model) / torch.nansum(targets) / math.log(2)


def compute_metrics(
    true_rates,
    inf_rates,
    true_latents,
    inf_latents,
    true_inputs,
    inf_inputs,
    true_spikes,
    n_heldin,
    device=None,
):
    if device is None:
        device = true_rates.device
    # Compute Rate R2
    rate_r2 = r2_score(true_rates, inf_rates, multioutput="variance_weighted")

    # Compute Input R2
    if inf_inputs is None or true_inputs is None:
        input_r2 = np.nan
    else:
        lm = LinearRegression()
        lm.fit(inf_inputs, true_inputs)
        true_inputs_pred = lm.predict(inf_inputs)
        input_r2 = r2_score(
            true_inputs, true_inputs_pred, multioutput="variance_weighted"
        )

    # Compute Latent R2
    lm = LinearRegression()
    lm.fit(true_latents, inf_latents)
    latent_pred_flat = lm.predict(true_latents)
    latent_r2 = r2_score(inf_latents, latent_pred_flat, multioutput="variance_weighted")

    bps = bits_per_spike(
        torch.tensor(np.log(inf_rates)).float(), torch.tensor(true_spikes).float()
    ).item()
    hi_bps = bits_per_spike(
        torch.tensor(np.log(inf_rates[:, :n_heldin])).float(),
        torch.tensor(true_spikes[:, :n_heldin]).float(),
    ).item()
    ho_bps = bits_per_spike(
        torch.tensor(np.log(inf_rates[:, n_heldin:])).float(),
        torch.tensor(true_spikes[:, n_heldin:]).float(),
    ).item()

    torch.set_grad_enabled(True)
    inf_latents_torch = torch.tensor(inf_latents).float().to(device)
    inf_rates_torch = torch.tensor(inf_rates).float().to(device)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(inf_rates.shape[1], 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, inf_latents.shape[1]),
    ).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        pred = mlp(inf_rates_torch)
        loss = criterion(pred, inf_latents_torch)
        loss.backward()
        optimizer.step()

    metric_dict = {
        "rate_r2": rate_r2,
        "input_r2": input_r2,
        "state_r2": latent_r2,
        "bps": bps,
        "hi_bps": hi_bps,
        "ho_bps": ho_bps,
    }
    noise_levels = np.linspace(0, 1, 6)
    for noise in noise_levels:
        noised_rates_flat = inf_rates_torch + torch.rand_like(inf_rates_torch) * noise
        latent_pred_flat = mlp(noised_rates_flat).detach().cpu().numpy()
        cycle_con_r2 = r2_score(
            inf_latents, latent_pred_flat, multioutput="variance_weighted"
        )
        metric_dict[f"cycle_con_{noise:.2f}_r2"] = cycle_con_r2
    return metric_dict
