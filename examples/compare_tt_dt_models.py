# %%
from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

# Load the analysis
tt_path_NODE = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240123_NBFF_NODE_Comparison/"
)
tt_path_GRU = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240123_NBFF_GRU_Comparison/"
)

dt_NODE2NODE = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240123_NBFF_NODE2NODE_Comparison/"
)

dt_NODE2GRU = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240123_NBFF_NODE2GRU_Comparison/"
)
dt_GRU2GRU = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240123_NBFF_GRU2GRU_Comparison/"
)
dt_GRU2NODE = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240123_NBFF_GRU2NODE_Comparison/"
)


tt_analysis_NODE = Analysis_TT(run_name="NBFF_TT_NODE", filepath=tt_path_NODE)
tt_analysis_GRU = Analysis_TT(run_name="NBFF_TT_GRU", filepath=tt_path_GRU)

dt_analysis_NODE2NODE = Analysis_DT(run_name="NBFF_DT_NODE2NODE", filepath=dt_NODE2NODE)
dt_analysis_NODE2GRU = Analysis_DT(run_name="NBFF_DT_NODE2GRU", filepath=dt_NODE2GRU)
dt_analysis_GRU2GRU = Analysis_DT(run_name="NBFF_DT_GRU2GRU", filepath=dt_GRU2GRU)
dt_analysis_GRU2NODE = Analysis_DT(run_name="NBFF_DT_GRU2NODE", filepath=dt_GRU2NODE)


# %%
comp = Comparison()
comp.load_analysis(tt_analysis_NODE)
comp.load_analysis(tt_analysis_GRU)

comp.load_analysis(dt_analysis_NODE2NODE)
comp.load_analysis(dt_analysis_NODE2GRU)
comp.load_analysis(dt_analysis_GRU2GRU)
comp.load_analysis(dt_analysis_GRU2NODE)

# comp.load_analysis(jslds_analysis)

# %%
comp.compare_latents_vaf()
comp.plot_trials(num_trials=2)

# %%
