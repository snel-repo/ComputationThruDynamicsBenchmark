import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

jsLDS_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/latents/jslds_outputs_3bff.h5"
)
with h5py.File(jsLDS_path, "r") as h5file:
    # Check the fields
    print(h5file.keys())
    lats = h5file["jslds_factors"][()]
    rates = h5file["jslds_rates"][()]
true_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/latents/"
    "3BFF_model_GRU_RNN_n_neurons_50_nonlin_embed_False_obs_noise_poisson_seed_0.h5"
)
with h5py.File(true_path, "r") as h5file:
    print(h5file.keys())
    train_lats = h5file["train_latents"][()]
    valid_lats = h5file["valid_latents"][()]
    train_rates = h5file["train_activity"][()]
    valid_rates = h5file["valid_activity"][()]

# Compute R2 between rates and latents
x = 1
true_lats = np.concatenate((train_lats, valid_lats), axis=0)
true_rates = np.concatenate((train_rates, valid_rates), axis=0)

# Compute R2 between true_latents and latents
B, T, D = true_lats.shape
true_lats_flat = true_lats.reshape(B * T, D)
B1, T1, D1 = lats.shape
lats_flat = lats.reshape(B1 * T1, D1)

pca = PCA()
pca.fit(true_lats_flat)
cum_var = np.cumsum(pca.explained_variance_ratio_)
top_n = np.where(cum_var > 0.99)[0][0] + 1

pca_jsLDS = PCA()
pcLats_jsLDS = pca_jsLDS.fit_transform(lats_flat)

true_lats_flat_pca = pca.transform(true_lats_flat)
true_lats_comp = true_lats_flat_pca

# Compute R2 between true_rates and rates
B, T, D = true_rates.shape
true_rates_flat = true_rates.reshape(B * T, D)
B1, T1, D1 = rates.shape
rates_flat = rates.reshape(B1 * T1, D1)
pred_true = np.zeros((B * T, top_n))

for i in range(top_n):
    reg = LinearRegression().fit(lats_flat, true_lats_comp[:, i])
    pred_true[:, i] = reg.predict(lats_flat)
    print(f"PC{i+1} R2: {reg.score(lats_flat, true_lats_comp[:,i])}")

for i in range(D):
    reg = LinearRegression().fit(rates_flat, true_rates_flat[:, i])
    print(f"Rate {i+1} R2: {reg.score(rates_flat, true_rates_flat[:,i])}")

# Plot the latents
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(
    true_lats_comp[1::100, 0],
    true_lats_comp[1::100, 1],
    true_lats_comp[1::100, 2],
    s=1,
    color="cyan",
    label="true",
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.scatter(
    pred_true[1::100, 0],
    pred_true[1::100, 1],
    pred_true[1::100, 2],
    s=1,
    color="g",
    label="pred",
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.scatter(
    pcLats_jsLDS[1::100, 0],
    pcLats_jsLDS[1::100, 1],
    pcLats_jsLDS[1::100, 2],
    s=1,
    color="r",
    label="jsLDS",
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

plt.savefig("jsLDS_latents.png")

rate_pca = PCA()
rates_dimRed = rate_pca.fit_transform(true_rates_flat)
jsLDS_rate_pca = PCA()
jsLDS_rates_dimRed = jsLDS_rate_pca.fit_transform(rates_flat)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(
    rates_dimRed[1::100, 0],
    rates_dimRed[1::100, 1],
    rates_dimRed[1::100, 2],
    s=1,
    color="cyan",
    label="true",
)
ax.set_xlabel("Rate PC1")
ax.set_ylabel("Rate PC2")
ax.set_zlabel("Rate PC3")

ax.legend()

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter(
    jsLDS_rates_dimRed[1::100, 0],
    jsLDS_rates_dimRed[1::100, 1],
    jsLDS_rates_dimRed[1::100, 2],
    s=1,
    color="r",
    label="jsLDS",
)
ax.set_xlabel("Rate PC1")
ax.set_ylabel("Rate PC2")
ax.set_zlabel("Rate PC3")

ax.legend()

plt.savefig("jsLDS_rates.png")
