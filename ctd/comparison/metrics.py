import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score


# TODO Make metrics agnostic to the analysis class
def get_rate_r2(rates_true, rates_pred):
    # Function to compare the rate-reconstruction of the different models
    if len(rates_pred.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = rates_pred.shape
        rates_pred_flat = (
            rates_pred.reshape(n_b_pred * n_t_pred, n_d_pred).detach().numpy()
        )
    else:
        rates_pred_flat = rates_pred.detach().numpy()

    if len(rates_true.shape) == 3:
        n_b_true, n_t_true, n_d_true = rates_true.shape
        rates_true_flat = (
            rates_true.reshape(n_b_true * n_t_true, n_d_true).detach().numpy()
        )
    else:
        rates_true_flat = rates_true.detach().numpy()
    # lr = LinearRegression().fit(rates_pred_flat, rates_true_flat)
    # preds = lr.predict(rates_pred_flat)
    # r2_rates = r2_score(rates_true_flat, preds, multioutput="variance_weighted")
    r2_rates = r2_score(rates_true_flat, rates_pred_flat)
    return r2_rates


def get_state_r2(lats_true, lats_pred, num_pcs=3):
    # Function to compare the latent activity821
    n_b_pred, n_t_pred, n_d_pred = lats_pred.shape
    lats_pred_flat = lats_pred.reshape(n_b_pred * n_t_pred, n_d_pred).detach().numpy()
    pca = PCA(n_components=num_pcs)
    if lats_pred_flat.shape[1] < num_pcs:
        # append zeros to the latent activity
        lats_pred_flat = np.concatenate(
            [
                lats_pred_flat,
                np.zeros((lats_pred_flat.shape[0], num_pcs - lats_pred_flat.shape[1])),
            ],
            axis=1,
        )
    lats_pred_flat_pca = pca.fit_transform(lats_pred_flat)
    lats_pred = lats_pred_flat_pca.reshape((n_b_pred, n_t_pred, num_pcs))

    n_b_true, n_t_true, n_d_true = lats_true.shape
    lats_true_flat = lats_true.reshape(n_b_true * n_t_true, n_d_true).detach().numpy()
    pca = PCA(n_components=num_pcs)
    lats_true_flat_pca = pca.fit_transform(lats_true_flat)
    lats_true = lats_true_flat_pca.reshape((n_b_true, n_t_true, num_pcs))

    # Compare the latent activity
    state_r2 = []
    for j in range(num_pcs):
        reg = LinearRegression().fit(lats_pred_flat_pca, lats_true_flat_pca[:, j])
        state_r2.append(reg.score(lats_pred_flat_pca, lats_true_flat_pca[:, j]))

    state_r2 = np.array(state_r2)
    return np.mean(state_r2)


def get_state_r2_vaf(lats_true, lats_pred):
    # Function to compare the latent activity
    if len(lats_pred.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = lats_pred.shape
        lats_pred_flat = (
            lats_pred.reshape(n_b_pred * n_t_pred, n_d_pred).detach().numpy()
        )
    else:
        lats_pred_flat = lats_pred.detach().numpy()

    if len(lats_true.shape) == 3:
        n_b_true, n_t_true, n_d_true = lats_true.shape
        lats_true_flat = (
            lats_true.reshape(n_b_true * n_t_true, n_d_true).detach().numpy()
        )
    else:
        lats_true_flat = lats_true.detach().numpy()

    # Compare the latent activity
    reg = LinearRegression().fit(lats_true_flat, lats_pred_flat)
    preds = reg.predict(lats_true_flat)
    state_r2 = r2_score(lats_pred_flat, preds, multioutput="variance_weighted")
    return state_r2


def get_latents_vaf(lats1, lats2, num_pcs=3):
    lats1_flat = (
        lats1.reshape(lats1.shape[0] * lats1.shape[1], lats1.shape[2]).detach().numpy()
    )
    pca = PCA(n_components=num_pcs)
    lats1_flat_pca = pca.fit_transform(lats1_flat)

    lats2_flat = (
        lats2.reshape(lats2.shape[0] * lats2.shape[1], lats2.shape[2]).detach().numpy()
    )
    pca = PCA(n_components=num_pcs)
    lats2_flat_pca = pca.fit_transform(lats2_flat)
    reg = LinearRegression().fit(lats1_flat_pca, lats2_flat_pca)
    preds = reg.predict(lats1_flat_pca)
    var_exp = explained_variance_score(
        lats2_flat_pca, preds, multioutput="variance_weighted"
    )
    return var_exp
