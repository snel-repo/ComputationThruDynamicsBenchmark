import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from interpretability.comparison.comparisons import Comparisons

# plt.ion()

# %matplotlib notebook
model_type = "NODE"
phase = "mem1"
task_to_analyze_1 = "MemoryPro"
task_to_analyze_2 = "MemoryAnti"

if model_type == "NODE":
    suffix = "MultiTaskNODE_Batch"
    filepath1 = (
        "/home/csverst/Github/InterpretabilityBenchmark/"
        "trained_models/task-trained/20231128_MT_NODE_5D_32Targets_FPFinding/"
    )
elif model_type == "GRU":
    suffix = "MultiTaskGRU_Batch"
    filepath1 = (
        "/home/csverst/Github/InterpretabilityBenchmark/"
        "trained_models/task-trained/20231128_MT_GRU_RNN_256D_32Targets_FPFinding/"
    )

plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    f"interpretability/comparison/plots/{suffix}/"
)
os.makedirs(plot_path, exist_ok=True)

comp = Comparisons(suffix=suffix)
comp.load_task_train_wrapper(filepath=filepath1)
tt_fps_pro, readout_pro = comp.compute_TT_FPs_MultiTask(
    task_to_analyze=task_to_analyze_1, phase=phase
)

tt_fps_anti, readout_anti = comp.compute_TT_FPs_MultiTask(
    task_to_analyze=task_to_analyze_2, phase=phase
)

x_pro = tt_fps_pro.xstar
x_anti = tt_fps_anti.xstar

x_both = np.concatenate((x_pro, x_anti), axis=0)
readout_both = np.concatenate((readout_pro, readout_anti), axis=0)

pca = PCA(n_components=2)
pca.fit(x_both)
x_pro_pca = pca.transform(x_pro)
x_anti_pca = pca.transform(x_anti)

plt.figure()
# add 3d subplot
ax = plt.subplot(111, projection="3d")
ax.scatter(x_pro_pca[:, 0], x_pro_pca[:, 1], readout_pro[:, 1], c="r", s=1, label="Pro")
ax.scatter(
    x_anti_pca[:, 0], x_anti_pca[:, 1], readout_anti[:, 1], c="b", s=1, label="Anti"
)

ax.set_zlim(-1.1, 1.1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Readout")
ax.legend()
plt.title("PCA (pro+anti) of Fixed Points")
plt.savefig(
    plot_path
    + f"PCA_FP_{task_to_analyze_1}_{task_to_analyze_2}_{model_type}_{phase}.png"
)

pca = PCA(n_components=2)
pca.fit(x_pro)
x_pro_pca = pca.transform(x_pro)
x_anti_pca = pca.transform(x_anti)
plt.figure()
# add 3d subplot
ax = plt.subplot(111, projection="3d")
ax.scatter(x_pro_pca[:, 0], x_pro_pca[:, 1], readout_pro[:, 1], c="r", s=1, label="Pro")
ax.scatter(
    x_anti_pca[:, 0], x_anti_pca[:, 1], readout_anti[:, 1], c="b", s=1, label="Anti"
)
ax.legend()

ax.set_zlim(-1.1, 1.1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Readout")
plt.title("PCA (pro) of Fixed Points")
plt.savefig(
    plot_path
    + f"PCAPro_FP_{task_to_analyze_1}_{task_to_analyze_2}_{model_type}_{phase}.png"
)

pca = PCA(n_components=2)
pca.fit(x_anti)
x_pro_pca = pca.transform(x_pro)
x_anti_pca = pca.transform(x_anti)
plt.figure()
# add 3d subplot
ax = plt.subplot(111, projection="3d")
ax.scatter(x_pro_pca[:, 0], x_pro_pca[:, 1], readout_pro[:, 1], c="r", s=1, label="Pro")
ax.scatter(
    x_anti_pca[:, 0], x_anti_pca[:, 1], readout_anti[:, 1], c="b", s=1, label="Anti"
)
ax.legend()
ax.set_zlim(-1.1, 1.1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Readout")
plt.title("PCA (anti) of Fixed Points")
plt.savefig(
    plot_path
    + f"PCAAnti_FP_{task_to_analyze_1}_{task_to_analyze_2}_{model_type}_{phase}.png"
)
