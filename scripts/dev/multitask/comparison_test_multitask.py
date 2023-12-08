# %%
import os

from interpretability.comparison.comparisons import Comparisons

# plt.ion()

# %matplotlib notebook
model_type = "GRU"
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

comp.compute_FPs_MultiTask()
# comp.interpolate_FPs_MultiTask()

# %%
