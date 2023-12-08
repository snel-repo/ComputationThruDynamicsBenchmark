import os

from interpretability.comparison.comparisons import Comparisons

# plt.ion()

# %matplotlib notebook
model_type = "NODE"
if model_type == "NODE":
    suffix = "MotorNetNODE"
    filepath1 = (
        "/home/csverst/Github/InterpretabilityBenchmark/"
        "trained_models/task-trained/20231207_RandomTarget_NODE_20D_v3/"
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
comp.plotHandKinematics()
