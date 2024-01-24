# %%
import os

import dotenv
import torch

from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH")

# %%
# Load the analysis
tt_path_NODE = TRAINED_MODEL_PATH + ("task-trained/20240123_NBFF_NODE_Comparison/")
tt_path_GRU = TRAINED_MODEL_PATH + ("task-trained/20240123_NBFF_GRU_Comparison/")

tt_analysis_NODE = Analysis_TT(run_name="NBFF_TT_NODE", filepath=tt_path_NODE)
tt_analysis_GRU = Analysis_TT(run_name="NBFF_TT_GRU", filepath=tt_path_GRU)

# %%
comp = Comparison()
comp.load_analysis(tt_analysis_NODE)
comp.load_analysis(tt_analysis_GRU)

# %%
input1 = torch.zeros(3)
tt_analysis_NODE.plot_fps(inputs=input1)
# %%
tt_analysis_GRU.plot_fps(inputs=input1)

# %%
