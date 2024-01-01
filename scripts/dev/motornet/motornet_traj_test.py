import os

from interpretability.comparison.analysis import Comparisons

# plt.ion()

# %matplotlib notebook
model_type = "GRU"
if model_type == "NODE":
    suffix = "MotorNetNODE"
    filepath1 = (
        "/home/csverst/Github/InterpretabilityBenchmark/"
        "trained_models/task-trained/20231207_RandomTarget_NODE_20D_v3/"
    )
elif model_type == "GRU":
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

comp = Comparisons(suffix=suffix)
comp.load_task_train_wrapper(filepath=filepath1)
# comp.plotHandKinematics()
# comp.plotPrepActivity()
# comp.plotPrepActivityLowPrep(prep_thresh = 10)
comp.generateVidPlanningSubspace()
