import os

# Import pca
import dotenv

from ctd.comparison.analysis.dt.dt import Analysis_DT
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(dotenv.find_dotenv())


HOME_DIR = os.environ["HOME_DIR"]
print(HOME_DIR)

pathTT = (
    HOME_DIR
    + "content/trained_models/task-trained/20240419_RandomTarget_"
    + "NoisyGRU_Final/max_epochs=1500 latent_size=64 seed=0/"
)
an_TT = Analysis_TT(run_name="TT", filepath=pathTT)


path_GRU_Sweep = pathTT + "20240422_Fig1_RandomTarget_GRU_Sweep2/"
subfolders_GRU = [f.path for f in os.scandir(path_GRU_Sweep) if f.is_dir()]

path_Vanilla_Sweep = pathTT + "20240419_Fig1_RandomTarget_Vanilla_Sweep/"
subfolders_Vanilla = [f.path for f in os.scandir(path_Vanilla_Sweep) if f.is_dir()]

path_NODE_Sweep = pathTT + "20240419_Fig1_RandomTarget_NODE_Sweep/"
subfolders_NODE = [f.path for f in os.scandir(path_NODE_Sweep) if f.is_dir()]

path_LFADS_Sweep = pathTT + "20240419_Fig1_RandomTarget_LFADS_Sweep/"
subfolders_LFADS = [f.path for f in os.scandir(path_LFADS_Sweep) if f.is_dir()]

comparison = Comparison(comparison_tag="Figure1RTR")
comparison.load_analysis(an_TT, reference_analysis=True, group="TT")

for subfolder in subfolders_LFADS:

    subfolder = subfolder + "/"
    analysis_temp = Analysis_DT(
        run_name="LFADS", filepath=subfolder, model_type="LFADS"
    )
    comparison.load_analysis(analysis_temp, group="LFADS")

comparison.regroup()

comparison.compare_state_rate_r2()
