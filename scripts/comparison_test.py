from interpretability.comparison.comparisons import Comparisons

plot_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/plots/"
)
comp_GRU2GRU = Comparisons(suffix="GRU2GRU")
comp_GRU2GRU.load_task_train_wrapper(
    filepath=(
        "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
        "20230921_NBFF_GRU_RNN_1000/"
    )
)
comp_GRU2GRU.load_data_train_wrapper(
    filepath=(
        "/home/csverst/Github/InterpretabilityBenchmark/trained_models/data-trained/"
        "20230922_3BFF_GRU_RNN_ExtInputs2/"
    )
)
# dict_g2g = comp_GRU2GRU.compareLatentActivity()
comp_GRU2GRU.computeFPs()
comp_GRU2GRU.plotTrial(0)
comp_GRU2GRU.plotLatentActivity()
comp_GRU2GRU.saveComparisonDict()
