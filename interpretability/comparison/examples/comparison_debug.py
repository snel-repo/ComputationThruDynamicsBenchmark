# %%
from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.external.ext import ExternalAnalysis
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

# Load the analysis

tt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240118_NBFF_GRU_CompareTest/"
)
dt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/data-trained/20240118_NBFF_GRU_CompareTest/"
)
lds_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/latents/3bff_jslds_with_inputs.h5"
)

tt_analysis = Analysis_TT(run_name="20240117_MultiTask_GRU", filepath=tt_path)
dt_analysis = Analysis_DT(
    run_name="20240118_3BFF_GRU_ExtInputs_CompTest", filepath=dt_path
)
jslds_analysis = ExternalAnalysis(run_name="3bff_jslds_with_inputs", filepath=lds_path)


# %%
comp = Comparison()
comp.load_analysis(tt_analysis)
comp.load_analysis(dt_analysis)
# comp.load_analysis(jslds_analysis)

# %%

comp.compare_latents()
comp.plot_trials(num_trials=2)
# %%

x = 1
