# %%

import os

import dotenv

from ctd.comparison.analysis.dt.dt import Analysis_DT
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("HOME_DIR") + "content/trained_models/"

fpath_tt = (
    TRAINED_MODEL_PATH
    + "task-trained/20240328_NBFF_GRU_Final/n=3 max_epochs=1500 seed=0/"
)
fpath_dt = fpath_tt + "20240328_GRU_RNN_DT_Final/"

tt_analysis = Analysis_TT(run_name="NBFF_TT", filepath=fpath_tt)
dt_analysis = Analysis_DT(run_name="NBFF_DT", filepath=fpath_dt, model_type="SAE")

tt_analysis.plot_trial_io(num_trials=1)
dt_analysis.plot_trial(num_trials=1, scatterPlot=False)
# %%
comp = Comparison()
comp.load_analysis(tt_analysis, reference_analysis=True)
comp.load_analysis(dt_analysis)

# %%
comp.plot_trials_reference(num_trials=2)
# %%
comp.compare_dynamics_DSA()

# %%
