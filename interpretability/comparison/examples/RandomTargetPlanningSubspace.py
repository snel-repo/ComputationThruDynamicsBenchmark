import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.visualizations.general import (
    animate_latent_trajectories,
)
from interpretability.comparison.visualizations.RandomTarget import plotPrepActivity

# plt.ion()

# %matplotlib notebook

suffix = "MotorNetGRU"
filepath1 = (
    "/home/csverst/Github/InterpretabilityBenchmark/trained_models/"
    "task-trained/20231220_RTDelay_RNN_valid/"
)

plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    f"interpretability/comparison/plots/{suffix}/"
)
os.makedirs(plot_path, exist_ok=True)


analysis_RT = Analysis_TT(run_name=suffix, filepath=filepath1)

tt_ics, tt_inputs, tt_targets = analysis_RT.get_model_input()

out_dict = analysis_RT.get_model_output()

latents = out_dict["latents"]
controlled = out_dict["controlled"]
actions = out_dict["actions"]

targets1 = tt_targets.detach().numpy()
targets = targets1[:, -1, :]
startPos = targets1[:, 0, :]

# find the index of the target appearing
target_on_inds = np.where(np.diff(tt_inputs[:, :, 0], axis=1) != 0)[1]
# find the index of the go cue
go_cue_inds = np.where(np.diff(tt_targets[:, :, 0], axis=1) != 0)[1]
prep_time = go_cue_inds - target_on_inds

# Get the snippet of the controlled trajectory
prep_latents = latents[:, go_cue_inds[0] - 2 : go_cue_inds[0], :]
prep_lats_mean = torch.mean(prep_latents, axis=1).detach().numpy()
B, T, D = latents.shape
pca = PCA(n_components=10)
prep_lats_mean = pca.fit_transform(prep_lats_mean)
lats_pca = pca.transform(latents.reshape(-1, D).detach().numpy())
lats_pca = lats_pca.reshape(B, T, -1)

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
latents = latents.detach().numpy()

startX_dict = {
    "vector": lrStartX.coef_,
    "label": "StartX",
}
startY_dict = {
    "vector": lrStartY.coef_,
    "label": "StartY",
}
targX_dict = {
    "vector": lrTargetX.coef_,
    "label": "TargetX",
}
targY_dict = {
    "vector": lrTargetY.coef_,
    "label": "TargetY",
}

pc1_vec = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
pc2_vec = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
pc3_vec = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

pc1_dict = {
    "vector": pc1_vec,
    "label": "PC1",
}
pc2_dict = {
    "vector": pc2_vec,
    "label": "PC2",
}
pc3_dict = {
    "vector": pc3_vec,
    "label": "PC3",
}

plotPrepActivity(analysis=analysis_RT)
animate_latent_trajectories(
    lats_pca,
    x_dict=targX_dict,
    y_dict=targY_dict,
    z_dict=startX_dict,
    filename="TargetOnDyn.mp4",
    align_idx=target_on_inds,
    align_str="target_on",
    pre_align=20,
    post_align=30,
    trail_length=2,
    trial_coloring=targets,
    suffix=suffix,
    elev=90,
    azim=-90,
)

animate_latent_trajectories(
    lats_pca,
    x_dict=targX_dict,
    y_dict=targY_dict,
    z_dict=startX_dict,
    filename="GoCueDyn.mp4",
    align_idx=go_cue_inds,
    align_str="go_cue",
    pre_align=20,
    post_align=30,
    trail_length=2,
    suffix=suffix,
    trial_coloring=targets,
)

animate_latent_trajectories(
    lats_pca,
    x_dict=pc1_dict,
    y_dict=pc2_dict,
    z_dict=pc3_dict,
    filename="TargetOnDynPCs.mp4",
    align_idx=target_on_inds,
    align_str="target_on",
    pre_align=20,
    post_align=30,
    trail_length=2,
    suffix=suffix,
    trial_coloring=targets,
)

animate_latent_trajectories(
    lats_pca,
    x_dict=pc1_dict,
    y_dict=pc2_dict,
    z_dict=pc3_dict,
    filename="GoCueDynPCs.mp4",
    align_idx=go_cue_inds,
    align_str="go_cue",
    pre_align=20,
    post_align=30,
    trail_length=2,
    suffix=suffix,
    trial_coloring=targets,
)


animate_latent_trajectories(
    lats_pca,
    x_dict=startX_dict,
    y_dict=startY_dict,
    z_dict=pc1_dict,
    filename="TargetOnDynStartPos.mp4",
    align_idx=target_on_inds,
    align_str="target_on",
    pre_align=20,
    post_align=30,
    trail_length=2,
    trial_coloring=startPos,
    suffix=suffix,
    elev=90,
    azim=-90,
)

animate_latent_trajectories(
    lats_pca,
    x_dict=startX_dict,
    y_dict=startY_dict,
    z_dict=pc1_dict,
    filename="GoCueDynStartPos.mp4",
    align_idx=go_cue_inds,
    align_str="go_cue",
    pre_align=20,
    post_align=50,
    trail_length=2,
    trial_coloring=startPos,
    plot_title="Colored by Start Position",
    elev=90,
    azim=-90,
    suffix=suffix,
)

animate_latent_trajectories(
    lats_pca,
    x_dict=startX_dict,
    y_dict=startY_dict,
    z_dict=pc1_dict,
    filename="GoCueDynStartPos_targ_pos.mp4",
    align_idx=go_cue_inds,
    align_str="go_cue",
    pre_align=20,
    post_align=50,
    trail_length=2,
    trial_coloring=targets,
    plot_title="Colored by Target Position",
    elev=90,
    azim=-90,
    suffix=suffix,
)
