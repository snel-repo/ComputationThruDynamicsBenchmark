import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ctd.comparison.analysis.tt.tasks.tt_MultiTask import Analysis_TT_MultiTask

# suffix = "MultiTaskGRU_LowLR"
# filepath1 = (
# "/home/csverst/Github/InterpretabilityBenchmark/"
# "trained_models/task-trained/20240116_MultiTask_GRU_lowLR/"
# )

suffix = "MultiTaskGRU_Final1"
filepath1 = (
    "/home/csverst/Github/InterpretabilityBenchmark/old/trained_models/task-trained/"
    "20240220_MultiTask_GRU_WeightDecay/max_epochs=200 seed=0/"
)
plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    f"interpretability/comparison/plots/{suffix}/"
)
os.makedirs(plot_path, exist_ok=True)


comp = Analysis_TT_MultiTask(run_name=suffix, filepath=filepath1)


# Get the model outputs
task1 = "MemoryPro"
task2 = "MemoryAnti"

ics, inputs, targets = comp.get_model_input()

out_dict = comp.get_model_output()
lats = out_dict["latents"]
outputs = out_dict["controlled"]

# Get the flag for the task (which trials are the correct task)
flag_pro, phase_task_pro = comp.get_task_flag(task1)
flag_anti, phase_task_anti = comp.get_task_flag(task2)

len_pro = [phase["response"][1] for phase in phase_task_pro]
len_anti = [phase["response"][1] for phase in phase_task_anti]

lats_pro = lats[flag_pro].detach().numpy()
lats_anti = lats[flag_anti].detach().numpy()
outputs_pro = outputs[flag_pro].detach().numpy()
outputs_anti = outputs[flag_anti].detach().numpy()

lats_phase_pro = comp.get_data_from_phase(phase_task_pro, "stim1", lats_pro)
lats_phase_pro_flat = np.concatenate(lats_phase_pro)

lats_phase_anti = comp.get_data_from_phase(phase_task_anti, "stim1", lats_anti)
lats_phase_anti_flat = np.concatenate(lats_phase_anti)

# Get PCA for both tasks
pca_pro = PCA()
pca_pro.fit(lats_phase_pro_flat)
pca_anti = PCA()
pca_anti.fit(lats_phase_anti_flat)

lats_pro_flat = lats_pro.reshape(
    lats_pro.shape[0] * lats_pro.shape[1], lats_pro.shape[2]
)
lats_anti_flat = lats_anti.reshape(
    lats_anti.shape[0] * lats_anti.shape[1], lats_anti.shape[2]
)

lats_pro_in_pro = pca_pro.transform(lats_pro_flat)
lats_anti_in_pro = pca_pro.transform(lats_anti_flat)
lats_pro_in_anti = pca_anti.transform(lats_pro_flat)
lats_anti_in_anti = pca_anti.transform(lats_anti_flat)

lats_pro_in_pro = lats_pro_in_pro.reshape(lats_pro.shape)
lats_anti_in_pro = lats_anti_in_pro.reshape(lats_anti.shape)
lats_pro_in_anti = lats_pro_in_anti.reshape(lats_pro.shape)
lats_anti_in_anti = lats_anti_in_anti.reshape(lats_anti.shape)

n_pro_trials = lats_pro.shape[0]
n_anti_trials = lats_anti.shape[0]

fig = plt.figure(figsize=(10, 10))
phases = phase_task_pro[0].keys()
for i, phase in enumerate(phases):

    ax1 = fig.add_subplot(4, 4, 4 * i + 1, projection="3d")
    for j in range(n_pro_trials):
        start_ind = phase_task_pro[j][phase][0]
        end_ind = phase_task_pro[j][phase][1]
        ax1.plot(
            lats_pro_in_pro[j, start_ind:end_ind, 0],
            lats_pro_in_pro[j, start_ind:end_ind, 1],
            outputs_pro[j, start_ind:end_ind, 1],
            color="blue",
            alpha=0.3,
        )
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([-3, 3])
    ax1.set_ylabel(phase)

    ax2 = fig.add_subplot(4, 4, 4 * i + 2, projection="3d")
    for j in range(n_anti_trials):
        start_ind = phase_task_anti[j][phase][0]
        end_ind = phase_task_anti[j][phase][1]
        ax2.plot(
            lats_anti_in_anti[j, start_ind:end_ind, 0],
            lats_anti_in_anti[j, start_ind:end_ind, 1],
            outputs_anti[j, start_ind:end_ind, 1],
            color="blue",
            alpha=0.3,
        )
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_zlim([-3, 3])

    start_ind = phase_task_pro[0][phase][0]
    end_ind = phase_task_pro[0][phase][1]
    ax3 = fig.add_subplot(4, 4, 4 * i + 3, projection="3d")
    for j in range(n_pro_trials):
        start_ind = phase_task_pro[j][phase][0]
        end_ind = phase_task_pro[j][phase][1]
        ax3.plot(
            lats_pro_in_anti[j, start_ind:end_ind, 0],
            lats_pro_in_anti[j, start_ind:end_ind, 1],
            outputs_pro[j, start_ind:end_ind, 1],
            color="blue",
            alpha=0.3,
        )
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-3, 3])
    ax3.set_zlim([-3, 3])

    ax4 = fig.add_subplot(4, 4, 4 * i + 4, projection="3d")
    for j in range(n_anti_trials):
        start_ind = phase_task_anti[j][phase][0]
        end_ind = phase_task_anti[j][phase][1]
        ax4.plot(
            lats_anti_in_pro[j, start_ind:end_ind, 0],
            lats_anti_in_pro[j, start_ind:end_ind, 1],
            outputs_anti[j, start_ind:end_ind, 1],
            color="blue",
            alpha=0.3,
        )
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 3])
    ax4.set_zlim([-3, 3])

    if i == 0:
        ax1.set_title("Pro")
        ax2.set_title("Anti")
        ax3.set_title("Pro in Anti")
        ax4.set_title("Anti in Pro")

plt.savefig(os.path.join(plot_path, f"PCA_plots{task1}_{task2}.png"))
plt.close(fig)
