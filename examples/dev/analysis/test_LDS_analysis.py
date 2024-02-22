# %%
import os

import dotenv

from interpretability.comparison.analysis.dt.dt import Analysis_DT

dotenv.load_dotenv(override=True)
TRAINED_MODEL_PATH = os.environ.get("HOME_DIR") + "data/trained_models/data-trained/"
fpath_sae = TRAINED_MODEL_PATH + "20240222_NBFF_SAE_test/"
fpath_jslds = TRAINED_MODEL_PATH + "20240222_NBFF_JSLDS_testing4/"
fpath_lfads = TRAINED_MODEL_PATH + "20240216_NBFF_LFADS_FPAnalysis/"

tt_sae = Analysis_DT(run_name="NBFF_SAE_test", filepath=fpath_sae, model_type="SAE")
tt_jslds = Analysis_DT(
    run_name="NBFF_JSLDS_testing4", filepath=fpath_jslds, model_type="LDS"
)
tt_lfads = Analysis_DT(
    run_name="NBFF_LFADS_FPAnalysis", filepath=fpath_lfads, model_type="LFADS"
)

# tt_sae.plot_trial(trial_num=0)
# tt_lfads.plot_trial(trial_num=0)
tt_jslds.plot_trial(trial_num=0)
# %%
