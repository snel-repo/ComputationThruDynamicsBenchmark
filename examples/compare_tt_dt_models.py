# %%
from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

# Load the analysis
tt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240123_NBFF_GRU_Comparison/"
)
dt_path_1 = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240123_NBFF_GRU_Comparison/"
)

tt_analysis = Analysis_TT(run_name="NBFF_TT_GRU", filepath=tt_path)
dt_analysis_1 = Analysis_DT(run_name="NBFF_DT_GRU", filepath=dt_path_1)


# %%
comp = Comparison()
comp.load_analysis(tt_analysis)
comp.load_analysis(dt_analysis_1)
# comp.load_analysis(jslds_analysis)

# %%
comp.compare_latents_vaf()
comp.plot_trials(num_trials=2)

# %%
