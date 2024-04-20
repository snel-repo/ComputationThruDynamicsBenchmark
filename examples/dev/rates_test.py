import os

import dotenv

from ctd.comparison.analysis.dt.dt import Analysis_DT
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(dotenv.find_dotenv())


HOME_DIR = os.environ["HOME_DIR"]
print(HOME_DIR)

pathTT = (
    HOME_DIR
    + "content/trained_models/task-trained/"
    + "20240328_NBFF_GRU_Final/n=3 max_epochs=1500 seed=0/"
)
pathGRU = pathTT + "20240328_GRU_RNN_DT_Final/"

pathLFADS = pathTT + "20240402_NBFF_LFADS_DT_WDecay_Sweep/"
pathLFADS1 = (
    pathLFADS
    + "gen_model=GRU_RNN prefix=20240328_NBFF_GRU_Final "
    + "seed=0 max_epochs=500 weight_decay=0.1/"
)
pathLFADS2 = (
    pathLFADS
    + "gen_model=GRU_RNN prefix=20240328_NBFF_GRU_Final"
    + "seed=0 max_epochs=500 weight_decay=0.001/"
)
pathLFADS3 = (
    pathLFADS
    + "gen_model=GRU_RNN prefix=20240328_NBFF_GRU_Final"
    + " seed=0 max_epochs=500 weight_decay=1e-05/"
)

an_TT = Analysis_TT(run_name="TT", filepath=pathTT)

an_GRU = Analysis_DT(run_name="GRU", filepath=pathGRU, model_type="SAE")
an_LFADS1 = Analysis_DT(run_name="1e-1", filepath=pathLFADS1, model_type="LFADS")
an_LFADS2 = Analysis_DT(run_name="1e-3", filepath=pathLFADS2, model_type="LFADS")
an_LFADS3 = Analysis_DT(run_name="1e-5", filepath=pathLFADS3, model_type="LFADS")

comparison = Comparison()
comparison.load_analysis(an_TT, reference_analysis=True)

comparison.load_analysis(an_GRU, group="GRU")
# comparison.load_analysis(an_LFADS1, group="LFADS")
# comparison.load_analysis(an_LFADS2, group="LFADS")
# comparison.load_analysis(an_LFADS3, group="LFADS")

comparison.compare_state_rate_r2()

sims = comparison.compare_dynamics_DSA()
