import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from interpretability.comparison.visualizations.general import PLOT_PATH


def plotPrepActivity(analysis):
    tt_ics, tt_inputs, tt_targets = analysis.get_model_input()
    out_dict = analysis.get_model_output()
    latents = out_dict["latents"]

    targets1 = tt_targets.detach().numpy()
    targets = targets1[:, -1, :]
    startPos = targets1[:, 0, :]
    # find the index of the switching target
    switch_inds = np.where(np.diff(tt_targets[:, :, 0], axis=1) != 0)[1]
    # find the index of the target appearing
    target_on_inds = np.where(np.diff(tt_inputs[:, :, 0], axis=1) != 0)[1]

    prep_time = switch_inds - target_on_inds
    # Get the snippet of the controlled trajectory
    # from 5 bins before the switch to 5 bins after
    prep_latents = latents[:, switch_inds[0] - 2 : switch_inds[0], :]
    prep_lats_mean = torch.mean(prep_latents, axis=1).detach().numpy()

    pca = PCA(n_components=10)
    prep_lats_mean = pca.fit_transform(prep_lats_mean)

    # Find axes in latent space that predict target position
    lrTargetX = LinearRegression().fit(prep_lats_mean, targets[:, 0])
    predTargX = lrTargetX.predict(prep_lats_mean)
    targR2X = lrTargetX.score(prep_lats_mean, targets[:, 0])
    lrTargetY = LinearRegression().fit(prep_lats_mean, targets[:, 1])
    predTargY = lrTargetY.predict(prep_lats_mean)
    targR2Y = lrTargetY.score(prep_lats_mean, targets[:, 1])

    lrStartX = LinearRegression().fit(prep_lats_mean, startPos[:, 0])
    predStartX = lrStartX.predict(prep_lats_mean)
    startR2X = lrStartX.score(prep_lats_mean, startPos[:, 0])
    lrStartY = LinearRegression().fit(prep_lats_mean, startPos[:, 1])
    predStartY = lrStartY.predict(prep_lats_mean)
    startR2Y = lrStartY.score(prep_lats_mean, startPos[:, 1])

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    # Scatter targets vs preds, opacity set by prep time
    prep_time_normalized = (prep_time - np.min(prep_time)) / (
        np.max(prep_time) - np.min(prep_time)
    )
    alphaVec = np.ones_like(prep_time_normalized)
    alphaVec[prep_time < 50] = 0.5
    ax.scatter(targets[:, 0], predTargX, color="r", alpha=alphaVec, label="X")
    ax.scatter(targets[:, 1], predTargY, color="b", alpha=alphaVec, label="Y")
    ax.text(0.75, 0.15, f"R2 x: {targR2X:.2f}", transform=ax.transAxes)
    ax.text(0.75, 0.08, f"R2 y: {targR2Y:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("Target Encoding")
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(startPos[:, 0], predStartX, color="r", label="X")
    ax.scatter(startPos[:, 1], predStartY, color="b", label="Y")
    ax.text(0.75, 0.15, f"R2 x: {startR2X:.2f}", transform=ax.transAxes)
    ax.text(0.75, 0.08, f"R2 y: {startR2Y:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Start")
    ax.set_ylabel("Predicted Start")
    ax.set_title("Start Encoding")
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(targets[:, 0], targets[:, 1], color="g", alpha=0.4, s=5, label="Target")
    ax.scatter(
        predTargX, predTargY, color="k", alpha=0.4, s=5, label="Predicted Target"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(startPos[:, 0], startPos[:, 1], color="y", alpha=0.4, s=5, label="Start")
    ax.scatter(
        predStartX, predStartY, color="k", alpha=0.4, s=5, label="Predicted Start"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # tight layout
    plt.tight_layout()
    plt.savefig(PLOT_PATH + analysis.run_name + "/" + "prep_TargetEncoding.png")

    lats_3D = prep_lats_mean[:, :3]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(lats_3D[:, 0], lats_3D[:, 1], lats_3D[:, 2], color="k", s=100)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig(PLOT_PATH + analysis.run_name + "/" + "prep_latents.png")

    w_startX = lrStartX.coef_
    w_startY = lrStartY.coef_
    w_targX = lrTargetX.coef_
    w_targY = lrTargetY.coef_

    # Project the weights onto each other to see if they are orthogonal
    w_startX = w_startX.reshape(1, -1)
    w_startY = w_startY.reshape(1, -1)
    w_targX = w_targX.reshape(1, -1)
    w_targY = w_targY.reshape(1, -1)

    w_startX = w_startX / np.linalg.norm(w_startX)
    w_startY = w_startY / np.linalg.norm(w_startY)
    w_targX = w_targX / np.linalg.norm(w_targX)
    w_targY = w_targY / np.linalg.norm(w_targY)

    startX_targX = np.dot(w_startX, w_targX.T)
    startX_targY = np.dot(w_startX, w_targY.T)
    startY_targX = np.dot(w_startY, w_targX.T)
    startY_targY = np.dot(w_startY, w_targY.T)

    print(f"StartX and TargX: {startX_targX}")
    print(f"StartX and TargY: {startX_targY}")
    print(f"StartY and TargX: {startY_targX}")
    print(f"StartY and TargY: {startY_targY}")
