# %%
import os

import dotenv

from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.analysis.external.ext import Analysis_Ext
from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH")

tt_path = TRAINED_MODEL_PATH + (
    "task-trained/20240202_NBFF_Comparison/"
    "latent_size=64 n_samples=1000 batch_size=512 max_epochs=1000 seed=0/"
)
dt_path = TRAINED_MODEL_PATH + ("data-trained/20240202_NBFF_Comparison/")
dt_path_2 = TRAINED_MODEL_PATH + ("data-trained/20240203_NBFF_Comparison/")
dt_path_3 = TRAINED_MODEL_PATH + ("data-trained/20240203_NBFF_Comparison_NODE/")
ext_path = TRAINED_MODEL_PATH + ("/external/3bff_jslds_with_inputs.h5")


tt_analysis = Analysis_TT(run_name="tt", filepath=tt_path)
dt_analysis = Analysis_DT(run_name="dt_128_GRU", filepath=dt_path)
dt_analysis_2 = Analysis_DT(run_name="dt_64_GRU", filepath=dt_path_2)
dt_analysis_3 = Analysis_DT(run_name="dt_NODE", filepath=dt_path_3)
ext_analysis = Analysis_Ext(run_name="ext", filepath=ext_path)

comp = Comparison()
comp.load_analysis(tt_analysis, reference_analysis=True)
comp.load_analysis(dt_analysis)
comp.load_analysis(dt_analysis_2)
comp.load_analysis(dt_analysis_3)
# comp.load_analysis(ext_analysis)

# comp.plot_trials(num_trials=2)
comp.plot_trials_reference(num_trials=2)
# %%
comp.compare_to_reference_affine()

# %%
comp.compare_state_r2()

# %%
