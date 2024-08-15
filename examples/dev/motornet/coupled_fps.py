# %%
import os

# Import pca
import dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle
from sklearn.decomposition import PCA

from ctd.comparison.analysis.tt.tasks.tt_RandomTarget import TT_RandomTarget
from ctd.task_modeling.datamodule.task_datamodule import TaskDataModule
from ctd.task_modeling.task_env.random_target import RandomTargetAligned

dotenv.load_dotenv(dotenv.find_dotenv())

HOME_DIR = os.environ["HOME_DIR"]
print(HOME_DIR)

pathTT = (
    "/home/csverst/Github/CtDBenchmark/pretrained/"
    + "20240605_RandomTarget_NoisyGRU_GoStep_ModL2_Delay/"
)

effector = RigidTendonArm26(muscle=MujocoHillMuscle())
an_TT = TT_RandomTarget(run_name="TT", filepath=pathTT)
wrapper = an_TT.wrapper

task = RandomTargetAligned(effector=effector, max_ep_duration=1.5)
# task.dataset_name = "MoveBump"

dm = TaskDataModule(task, n_samples=1100, batch_size=1000)
dm.set_environment(task, for_sim=True)
wrapper.set_environment(task)
an_TT.wrapper = wrapper
dm.prepare_data()
dm.setup()

an_TT.env = task
an_TT.datamodule = dm
# %%

fps = an_TT.compute_coupled_FPs(
    max_iters=100,
    learning_rate=0.005,
)
# %%
qvals = fps.qstar
xstar = fps.xstar

xstar = xstar[:, 14:]
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.hist(qvals, bins=100)

pca_fp = PCA(n_components=3)
pca_fp.fit(xstar)
xstar_pcs = pca_fp.transform(xstar)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xstar_pcs[:, 0], xstar_pcs[:, 1], xstar_pcs[:, 2])

# %%
self = an_TT
tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
inputs_to_env = self.get_inputs_to_env()
print(tt_inputs.shape)
tt_extra = self.get_extra_inputs()
out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env=inputs_to_env)
controlled = out_dict["controlled"]
states = out_dict["states"]
latents = out_dict["latents"]


pca = PCA(n_components=3)
lats_pcs = pca.fit_transform(latents.detach().numpy().reshape(-1, latents.shape[-1]))
lats_pcs = lats_pcs.reshape(latents.shape[0], latents.shape[1], -1)
xstar_latPC = pca.transform(xstar)
# %%
# Create scatter plot for xstar_latPC
scatter = go.Scatter3d(
    x=xstar_latPC[:, 0],
    y=xstar_latPC[:, 1],
    z=xstar_latPC[:, 2],
    mode="markers",
    marker=dict(
        size=2,
        # Make it an X
        symbol="x",
    ),
    name="Fixed Points",
)

start_scatter = go.Scatter3d(
    x=lats_pcs[:300, 0, 0],
    y=lats_pcs[:300, 0, 1],
    z=lats_pcs[:300, 0, 2],
    mode="markers",
    marker=dict(size=2, color="green"),
    name="TrialStart",
)

end_scatter = go.Scatter3d(
    x=lats_pcs[:300, -1, 0],
    y=lats_pcs[:300, -1, 1],
    z=lats_pcs[:300, -1, 2],
    mode="markers",
    marker=dict(size=2, color="red"),
    name="TrialEnd",
)

# Create lines for lats_pcs with green start markers and red end markers
lines = []
for i in range(300):
    line = go.Scatter3d(
        x=lats_pcs[i, :, 0],
        y=lats_pcs[i, :, 1],
        z=lats_pcs[i, :, 2],
        mode="lines",
        line=dict(color="black", width=0.5),
        showlegend=(i == 0),
        name="Latent activity",
    )
    lines.append(line)

# Combine scatter and lines
data = [scatter] + lines + [start_scatter] + [end_scatter]

# Set up the layout
layout = go.Layout(
    scene=dict(
        xaxis_title="Latent PC1", yaxis_title="Latent PC2", zaxis_title="Latent PC3"
    ),
    width=700,
    height=700,
    legend=dict(itemsizing="constant"),
)

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Show the plot
fig.show()

# %%
