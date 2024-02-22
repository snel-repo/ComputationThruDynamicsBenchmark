# %%


from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.tt.tt import Analysis_TT

tt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/data/"
    "trained_models/task-trained/20240216_NBFF_GRU_RNN_Final/seed=0/"
)
dt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/data/"
    "trained_models/data-trained/20240216_NBFF_LFADS_FPAnalysis/"
)
dt_path_GRU = (
    "/home/csverst/Github/InterpretabilityBenchmark/data/"
    "trained_models/data-trained/20240216_NBFF_GRU_FPAnalysis/"
)

tt_analysis = Analysis_TT("20240216_NBFF_GRU_RNN_Final", tt_path)
dt_analysis = Analysis_DT("20240216_NBFF_LFADS_FPAnalysis", dt_path, model_type="LFADS")
dt_analysis_SAE = Analysis_DT(
    "20240216_NBFF_GRU_FPAnalysis", dt_path_GRU, model_type="SAE"
)

tt_analysis.plot_fps()
dt_analysis_SAE.plot_fps()
dt_analysis.plot_fps()


# %%
