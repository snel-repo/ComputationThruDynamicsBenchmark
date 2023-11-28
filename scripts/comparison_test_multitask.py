# %%
from interpretability.comparison.comparisons import Comparisons

# plt.ion()

# %matplotlib notebook

plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/plots/"
)

comp = Comparisons(suffix="MultiTaskNODE")
comp.load_task_train_wrapper(
    filepath=(
        (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            "trained_models/task-trained/20231027_MultiTask_NODE_10D_32Targets/"
        )
    )
)
# comp = Comparisons(suffix="MultiTaskRNN")
# comp.load_task_train_wrapper(
#     filepath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/20231027_MultiTask_GRU_RNN_256D_32Targets/"
#     )
# )

# comp = Comparisons(suffix="MultiTaskRNNTest")
# comp.load_task_train_wrapper(
#     filepath=(
#         "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/20231025_MultiTask_GRU_RNN_256D_32Targets_10Epochs/"
#     )
# )
comp.compute_FPs_MultiTask()
# comp.interpolate_FPs_MultiTask()
