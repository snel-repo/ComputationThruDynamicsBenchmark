# %%
from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

jsLDS_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/trained_models/"
    "data-trained/20240202_NBFF_Comparison/"
)

tt_path = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "trained_models/task-trained/20240202_DSA_GRU_Comparison/"
    "latent_size=64 n_samples=1000 batch_size=512 max_epochs=1000 n=3 seed=0/"
)


dt_analysis = Analysis_DT(run_name="dt_NBFF", filepath=jsLDS_path)
tt_analysis = Analysis_TT(run_name="tt_NBFF", filepath=tt_path)

comp = Comparison()
comp.load_analysis(tt_analysis)
comp.load_analysis(dt_analysis)
comp.plot_trials(num_trials=2)

# %%
temp = comp.get_state_r2(dt_analysis, tt_analysis)
# %%
comp.compare_dynamics_DSA(n_delays=20, rank=50)
comp.plot_state_rate_R2()
