# %%
import os

import dotenv

from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH")
# %%
# Load the analysis
tt_path_NODE = TRAINED_MODEL_PATH + ("task-trained/20240123_NBFF_NODE_Comparison/")

tt_path_GRU = TRAINED_MODEL_PATH + ("task-trained/20240123_NBFF_GRU_Comparison/")

dt_NODE2NODE = TRAINED_MODEL_PATH + ("data-trained/20240123_NBFF_NODE2NODE_Comparison/")
dt_NODE2GRU = TRAINED_MODEL_PATH + ("data-trained/20240123_NBFF_NODE2GRU_Comparison/")
dt_GRU2GRU = TRAINED_MODEL_PATH + ("data-trained/20240123_NBFF_GRU2GRU_Comparison/")
dt_GRU2NODE = TRAINED_MODEL_PATH + ("data-trained/20240123_NBFF_GRU2NODE_Comparison/")


tt_analysis_NODE = Analysis_TT(run_name="NBFF_TT_NODE", filepath=tt_path_NODE)
tt_analysis_GRU = Analysis_TT(run_name="NBFF_TT_GRU", filepath=tt_path_GRU)

dt_analysis_NODE2NODE = Analysis_DT(run_name="NBFF_DT_NODE2NODE", filepath=dt_NODE2NODE)
dt_analysis_NODE2GRU = Analysis_DT(run_name="NBFF_DT_NODE2GRU", filepath=dt_NODE2GRU)
dt_analysis_GRU2GRU = Analysis_DT(run_name="NBFF_DT_GRU2GRU", filepath=dt_GRU2GRU)
dt_analysis_GRU2NODE = Analysis_DT(run_name="NBFF_DT_GRU2NODE", filepath=dt_GRU2NODE)


# %%
comp = Comparison()
comp.load_analysis(tt_analysis_NODE)
comp.load_analysis(dt_analysis_NODE2NODE)

comp.load_analysis(tt_analysis_GRU)
comp.load_analysis(dt_analysis_GRU2GRU)

comp.load_analysis(dt_analysis_GRU2NODE)
comp.load_analysis(dt_analysis_NODE2GRU)

# %%

comp.compare_dynamics_DSA(n_delays=20, rank=50)

# %%
comp.compare_latents_vaf()
comp.plot_trials(num_trials=2)

# %%
