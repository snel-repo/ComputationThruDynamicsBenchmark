# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from interpretability.comparison.comparisons import Comparisons

# plt.ion()

# %matplotlib notebook

plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/plots/"
)
# comp_GRU2GRU = Comparisons(suffix="GRU2GRU")
# comp_GRU2GRU.load_task_train_wrapper(
#     filepath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
#         "20230921_NBFF_GRU_RNN_1000/"
#     )
# )
# comp_GRU2GRU.load_data_train_wrapper(
#     filepath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/trained_models/data-trained/"
#         "20230922_3BFF_GRU_RNN_ExtInputs2/"
#     )
# )
# comp = comp_GRU2GRU

comp_NODE2GRU = Comparisons(suffix="NODE2GRU")
comp_NODE2GRU.load_task_train_wrapper(
    filepath=(
        "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
        "20230921_NBFF_GRU_RNN_1000/"
    )
)
comp_NODE2GRU.load_data_train_wrapper(
    filepath=(
        "/home/csverst/Github/InterpretabilityBenchmark/trained_models/data-trained/"
        "20230927_3BFF_NODE_ExtInputs/"
    )
)
comp = comp_NODE2GRU

dict2 = comp.compareLatentActivity()
# comp.plotTrial(0)
tt_db, dt_db = comp.compute_data_for_levelset()

comp.computeFPs()
comp.plotLatentActivity()
comp.compareLatentActivity()

comp.saveComparisonDict()

# %%

tt_fps = comp.tt_fps
dt_fps = comp.dt_fps
tt_fp_loc = tt_fps.xstar
dt_fp_loc = dt_fps.xstar

comp.tt_model.eval()
tt_train_ds = comp.tt_datamodule.train_ds
tt_val_ds = comp.tt_datamodule.valid_ds
tt_ics = torch.vstack((tt_train_ds.tensors[0], tt_val_ds.tensors[0]))
tt_inputs = torch.vstack((tt_train_ds.tensors[1], tt_val_ds.tensors[1]))
tt_model_out, tt_latents, tt_actions = comp.task_train_wrapper(tt_ics, tt_inputs)
dt_train_ds = comp.dt_datamodule.train_ds
dt_val_ds = comp.dt_datamodule.valid_ds
dt_train_inds = dt_train_ds.tensors[4]
dt_val_inds = dt_val_ds.tensors[4]
# transform to int
dt_train_inds = dt_train_inds.type(torch.int64)
dt_val_inds = dt_val_inds.type(torch.int64)

tt_latents_val = tt_latents[dt_val_inds]
tt_model_out_val = tt_model_out[dt_val_inds]
tt_latents_val = tt_latents_val.detach().numpy()
tt_model_out_val = tt_model_out_val.detach().numpy()
dt_spikes = dt_val_ds.tensors[0]
dt_inputs = dt_val_ds.tensors[2]
dt_log_rates, dt_latents = comp.dt_model(dt_spikes, dt_inputs)
dt_latents = dt_latents.detach().numpy()
dt_log_rates = dt_log_rates.detach().numpy()

tt_lats_flat = tt_latents_val.reshape(-1, tt_latents_val.shape[-1])
dt_lats_flat = dt_latents.reshape(-1, dt_latents.shape[-1])

tt_PCA = PCA().fit(tt_lats_flat)
dt_PCA = PCA().fit(dt_lats_flat)

tt_lats_flat_PCA = tt_PCA.transform(tt_lats_flat)
dt_lats_flat_PCA = dt_PCA.transform(dt_lats_flat)

tt2dt = LinearRegression().fit(tt_lats_flat_PCA, dt_lats_flat_PCA)
dt2tt = LinearRegression().fit(dt_lats_flat_PCA, tt_lats_flat_PCA)

tt_fp_loc = tt_fps.xstar
tt_fp_loc = tt_PCA.transform(tt_fp_loc)
tt_fp_loc_in_dt = tt2dt.predict(tt_fp_loc)

dt_fp_loc = dt_fps.xstar
dt_fp_loc = dt_PCA.transform(dt_fp_loc)

# %%


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection="3d")

ax.scatter(
    dt_fp_loc[:, 0],
    dt_fp_loc[:, 1],
    dt_fp_loc[:, 2],
    color="k",
    s=100,
    alpha=0.5,
    label="dt",
)
ax.scatter(
    tt_fp_loc_in_dt[:, 0],
    tt_fp_loc_in_dt[:, 1],
    tt_fp_loc_in_dt[:, 2],
    color="b",
    s=100,
    alpha=0.5,
    label="tt in dt",
)
# Get Pairwise distances
dists = np.zeros((dt_fp_loc.shape[0], tt_fp_loc_in_dt.shape[0]))
for i in range(dt_fp_loc.shape[0]):
    for j in range(tt_fp_loc_in_dt.shape[0]):
        dt_fp = dt_fp_loc[i]
        tt_fp = tt_fp_loc_in_dt[j]
        dists[i, j] = np.linalg.norm(dt_fp - tt_fp)
    min_dist = np.min(dists[i])
    min_dist_ind = np.argmin(dists[i])


matched_points_space1 = set()
matched_points_space2 = set()
threshold = 0.5
pairs = []
tt_pair = []
dt_pair = []
while dists.size > 0:
    # Find the minimum distance and its index
    min_distance = np.min(dists)
    if min_distance > threshold:
        break

    i, j = np.unravel_index(np.argmin(dists), dists.shape)

    if i not in matched_points_space1 and j not in matched_points_space2:
        matched_points_space1.add(i)
        matched_points_space2.add(j)

        # Draw red line between matched points
        ax.plot(
            [dt_fp_loc[i, 0], tt_fp_loc_in_dt[j, 0]],
            [dt_fp_loc[i, 1], tt_fp_loc_in_dt[j, 1]],
            [dt_fp_loc[i, 2], tt_fp_loc_in_dt[j, 2]],
            "r",
            alpha=1,
            linewidth=5,
        )

        # Remove matched points from the distance matrix
        dists[i, :] = float("inf")
        dists[:, j] = float("inf")
        pairs.append((i, j))
        tt_pair.append(tt_fp_loc_in_dt[j, :])
        dt_pair.append(dt_fp_loc[i, :])

tt_pair = np.array(tt_pair)
dt_pair = np.array(dt_pair)
lr_FP = LinearRegression().fit(tt_pair, dt_pair)
r2_FP = lr_FP.score(tt_pair, dt_pair)

# Plot some trajectories
dt_traj = dt_latents[0:1000:10]
dt_traj_flat = dt_traj.reshape(-1, dt_traj.shape[-1])
dt_traj_flat = dt_PCA.transform(dt_traj_flat)
dt_traj = dt_traj_flat.reshape(dt_traj.shape[0], -1, dt_traj.shape[-1])
for i in range(dt_traj.shape[0]):
    ax.plot(dt_traj[i, :, 0], dt_traj[i, :, 1], dt_traj[i, :, 2], alpha=0.1)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"NODE2GRU Matched FP R2: {r2_FP:.3f}")

plt.legend()
plt.show()
plt.savefig(plot_path + "fp_match.png")

plt.savefig(plot_path + "fp_match.pdf")
# ax.legend()


# comp.compareFPs()
# comp_dict = Comparisons.compareLatentActivityPath(
#     dtLatentsPath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/"
#         "interpretability/comparison/latents/GRU2GRU_dt_latents.pkl"
#     ),
#     ttLatentsPath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/"
#         "interpretability/comparison/latents/GRU2GRU_tt_latents.pkl"
#     ),
# )

# %%

# Your data and computations preceding the plot creation should be here...

# Create a list to hold the traces
traces = []

# Scatter plot for 'dt'
trace_dt = go.Scatter3d(
    x=dt_fp_loc[:, 0],
    y=dt_fp_loc[:, 1],
    z=dt_fp_loc[:, 2],
    mode="markers",
    marker=dict(size=5, color="black", opacity=0.5),
    name="dt",
)
traces.append(trace_dt)

# Scatter plot for 'tt in dt'
trace_tt = go.Scatter3d(
    x=tt_fp_loc_in_dt[:, 0],
    y=tt_fp_loc_in_dt[:, 1],
    z=tt_fp_loc_in_dt[:, 2],
    mode="markers",
    marker=dict(size=5, color="blue", opacity=0.5),
    name="tt in dt",
)
traces.append(trace_tt)

# Lines between matched points
for i, j in pairs:
    trace_line = go.Scatter3d(
        x=[dt_fp_loc[i, 0], tt_fp_loc_in_dt[j, 0]],
        y=[dt_fp_loc[i, 1], tt_fp_loc_in_dt[j, 1]],
        z=[dt_fp_loc[i, 2], tt_fp_loc_in_dt[j, 2]],
        mode="lines",
        line=dict(color="red", width=2),
        showlegend=False,
    )
    traces.append(trace_line)

# Trajectories
for i in range(dt_traj.shape[0]):
    trace_traj = go.Scatter3d(
        x=dt_traj[i, :, 0],
        y=dt_traj[i, :, 1],
        z=dt_traj[i, :, 2],
        mode="lines",
        line=dict(color="gray", width=1, dash="dashdot"),
        opacity=0.1,
        showlegend=False,
    )
    traces.append(trace_traj)

# Create a figure using the traces
fig = go.Figure(data=traces)

# Update layout
fig.update_layout(
    scene_xaxis_title="PC1",
    scene_yaxis_title="PC2",
    scene_zaxis_title="PC3",
    title=f"NODE2GRU Matched FP R2: {r2_FP:.3f}",
)

# Show the figure
fig.show()

# Save the plot
fig.write_image(plot_path + f"{comp.suffix}_fp_match.png")
fig.write_image(plot_path + f"{comp.suffix}_fp_match.pdf")

fig = go.Figure(data=traces)

frames = []

for t in np.linspace(0, 360, 100):  # Creating 100 frames
    frames.append(
        go.Frame(
            layout=go.Layout(
                scene_camera_eye=dict(x=np.cos(np.radians(t)), y=np.sin(np.radians(t)))
            )
        )
    )

fig.frames = frames

fig.update_layout(
    scene_xaxis_title="PC1",
    scene_yaxis_title="PC2",
    scene_zaxis_title="PC3",
    title=f"NODE2GRU Matched FP R2: {r2_FP:.3f}",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(frame=dict(duration=50, redraw=True), fromcurrent=True),
                    ],
                )
            ],
        )
    ],
)

fig.show()

# Save as HTML
fig.write_html(plot_path + f"{comp.suffix}_fp_match.html")

# If you need static images
fig.write_image(plot_path + f"{comp.suffix}_fp_match.png")
fig.write_image(plot_path + f"{comp.suffix}_fp_match.pdf")

# %%

# open the saved data h5 file and unpack the TT and DT tensors:
data_path = "/snel/share/runs/fpfinding/levelset_data/"
DT_filename = "GRU2GRU_dt_datablock.h5"
TT_filename = "GRU2GRU_tt_datablock.h5"

# load the DT data from h5 file:
print(f"{data_path + DT_filename}")
path1 = data_path + DT_filename
with h5py.File(path1, "r") as f:
    dt_data = f["dt_datablock"][()]
    dt_data = dt_data[0]
# %%

levelset_path = "/snel/share/runs/fpfinding/levelset_data/"
# Load the data from the h5 file
tt_db = h5py.File(levelset_path + "NODE2GRU_tt_datablock.h5", "r")
# Get the data from the h5 file
tt_db = tt_db["tt_datablock"][()]

dt_db = h5py.File(levelset_path + "NODE2GRU_dt_datablock.h5", "r")
# Get the data from the h5 file
dt_db = dt_db["dt_datablock"][()]
# %%
