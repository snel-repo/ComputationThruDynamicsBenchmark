# %%
import os

# Import pca
import dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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
    HOME_DIR
    + "content/trained_models/task-trained/20240527_RandomTarget_GRU_RNN_ShorterSim2/"
    + "max_epochs=1500 latent_size=128 seed=0 learning_rate=0.001/"
)
effector = RigidTendonArm26(muscle=MujocoHillMuscle())
an_TT = TT_RandomTarget(run_name="TT", filepath=pathTT)
wrapper = an_TT.wrapper

task = RandomTargetAligned(effector=effector, max_ep_duration=1.5)
task.dataset_name = "MoveBump"

dm = TaskDataModule(task, n_samples=1100, batch_size=1000)
dm.set_environment(task, for_sim=True)
wrapper.set_environment(task)
an_TT.wrapper = wrapper
dm.prepare_data()
dm.setup()

an_TT.env = task
an_TT.datamodule = dm
# %%
an_TT.plot_bump_response()
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
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
pca = PCA(n_components=3)
lats_pcs = pca.fit_transform(latents.detach().numpy().reshape(-1, latents.shape[-1]))
lats_pcs = lats_pcs.reshape(latents.shape[0], latents.shape[1], -1)


# %%
# Add a vertical line at the go cue
target_on_inds = tt_extra[:, 0].detach().numpy()
go_cue_inds = tt_extra[:, 1].detach().numpy()
states = states.detach().numpy()
bump_states = states[:, 65:70, :]
bump_states = bump_states.reshape(-1, bump_states.shape[-1])
pca = PCA(n_components=3)
bump_pca = pca.fit_transform(bump_states)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
inputs_bump = inputs_to_env[:, 65, :]
bump_ang = np.arctan2(inputs_bump[:, 1], inputs_bump[:, 0])
bump_type = np.unique(bump_ang, axis=0)
# Map bump type to color
bump_colors = cm.jet(np.linspace(0, 1, bump_type.shape[0]))
bump_ind = np.where(bump_ang == bump_type[0])[0]
for i in range(states.shape[0]):
    print(f"Bump: {bump_ang[i]}")
    ax.plot(
        bump_pca[i, 0],
        bump_pca[i, 1],
        bump_pca[i, 2],
        color=bump_colors[bump_ind],
    )
