# %%
import os

import dotenv

from interpretability.comparison.analysis.tt.tt import Analysis_TT
from interpretability.comparison.comparison import Comparison

dotenv.load_dotenv(override=True)
HOME_DIR = os.getenv("HOME_DIR")
task_trained_path = HOME_DIR + "trained_models/task-trained/20240211_NBFF_DSA_Test/"
# %%
# Load the analysis
tt_GRU = task_trained_path
print(tt_GRU)
subdirs = [x[0] for x in os.walk(tt_GRU)]
print(subdirs)
analysis_list = []
for subdir in subdirs[1:]:
    print(subdir)
    analysis_list.append(
        Analysis_TT(run_name=subdir.split("/")[-1], filepath=subdir + "/")
    )
rank_sweep = [100, 200]
delay_sweep = [4, 8]

analysis1 = analysis_list[0]
id_comp, splits_comp = analysis1.find_DSA_hps(
    rank_sweep=rank_sweep, delay_sweep=delay_sweep
)


comp = Comparison()
for analysis in analysis_list:
    comp.load_analysis(analysis)

comp.compare_dynamics_DSA(n_delays=16, rank=400)

# %%
