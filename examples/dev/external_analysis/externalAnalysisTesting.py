# %%
from interpretability.comparison.analysis.external.ext import Analysis_Ext
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

jsLDS_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/comparison/latents/3bff_jslds_with_inputs.h5"
)

tt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240202_DSA_GRU_Comparison/"
    "latent_size=64 n_samples=1000 batch_size=512 max_epochs=1000 n=3 seed=0/"
)


ext_analysis = Analysis_Ext(run_name="jsLDS", filepath=jsLDS_path)
tt_analysis = Analysis_TT(run_name="tt_NBFF", filepath=tt_path)

comp = Comparison()
comp.load_analysis(ext_analysis)
comp.load_analysis(tt_analysis)

comp.plot_trials(num_trials=2)

# %%
comp.compare_dynamics_DSA(n_delays=20, rank=50)
