# %%
import os

import dotenv

from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH")
# %%
# Load the analysis
tt_path_3BFF_NODE_1 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_3BFF_NODE_DSA_0_Longer/"
)
tt_path_3BFF_GRU_1 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_3BFF_GRU_DSA_0_Longer/"
)

tt_path_4BFF_NODE_1 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_4BFF_NODE_DSA_0_Longer/"
)
tt_path_4BFF_GRU_1 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_4BFF_GRU_DSA_0_Longer/"
)

tt_path_3BFF_NODE_2 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_3BFF_NODE_DSA_1_Longer/"
)
tt_path_3BFF_GRU_2 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_3BFF_GRU_DSA_1_Longer/"
)

tt_path_4BFF_NODE_2 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_4BFF_NODE_DSA_1_Longer/"
)
tt_path_4BFF_GRU_2 = TRAINED_MODEL_PATH + (
    "task-trained/20240125_4BFF_GRU_DSA_1_Longer/"
)


tt_analysis_3BFF_NODE = Analysis_TT(
    run_name="3BFF_NODE_0", filepath=tt_path_3BFF_NODE_1
)
tt_analysis_3BFF_GRU = Analysis_TT(run_name="3BFF_GRU_0", filepath=tt_path_3BFF_GRU_1)

tt_analysis_4BFF_NODE = Analysis_TT(
    run_name="4BFF_NODE_0", filepath=tt_path_4BFF_NODE_1
)
tt_analysis_4BFF_GRU = Analysis_TT(run_name="4BFF_GRU_0", filepath=tt_path_4BFF_GRU_1)

tt_analysis_3BFF_NODE_2 = Analysis_TT(
    run_name="3BFF_NODE_1", filepath=tt_path_3BFF_NODE_2
)
tt_analysis_3BFF_GRU_2 = Analysis_TT(run_name="3BFF_GRU_1", filepath=tt_path_3BFF_GRU_2)

tt_analysis_4BFF_NODE_2 = Analysis_TT(
    run_name="4BFF_NODE_1", filepath=tt_path_4BFF_NODE_2
)
tt_analysis_4BFF_GRU_2 = Analysis_TT(run_name="4BFF_GRU_1", filepath=tt_path_4BFF_GRU_2)


# %%
comp = Comparison()
comp.load_analysis(tt_analysis_3BFF_NODE)
comp.load_analysis(tt_analysis_3BFF_NODE_2)

comp.load_analysis(tt_analysis_3BFF_GRU)
comp.load_analysis(tt_analysis_3BFF_GRU_2)

comp.load_analysis(tt_analysis_4BFF_NODE)
comp.load_analysis(tt_analysis_4BFF_NODE_2)

comp.load_analysis(tt_analysis_4BFF_GRU)
comp.load_analysis(tt_analysis_4BFF_GRU_2)


# %%

comp.compare_dynamics_DSA(n_delays=20, rank=50)

# %%
comp.compare_latents_vaf()
comp.plot_trials(num_trials=2)

# %%
