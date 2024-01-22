import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from interpretability.comparison.analysis.analysis_tt import Comparisons

# plt.ion()

# %matplotlib notebook
model_type = "NODE"
task_to_analyze_1 = "MemoryPro"

phases = ["context", "stim1", "mem1", "response"]
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

plt.figure()
ax = plt.subplot(111, projection="3d")
colors = ["r", "b", "g", "k"]
for i, phase in enumerate(phases):
    tt_fps, readout = comp.compute_TT_FPs_MultiTask(
        task_to_analyze=task_to_analyze_1, phase=phase
    )
    x_star = tt_fps.xstar

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_star)

    # add 3d subplot
    # set fig as current figure
    ax.scatter(x_pca[:, 0], x_pca[:, 1], readout[:, 1], c=colors[i], s=4, label=phase)

    ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Readout")

ax.legend()
plt.savefig(plot_path + f"DriscollFig1_{task_to_analyze_1}_{model_type}.png")
