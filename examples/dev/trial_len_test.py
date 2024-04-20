# %%
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
    + "content/trained_models/task-trained/20240411_MultiTask_GRU_L2Lats/"
    + "max_epochs=500 latent_l2_wt=1e-07 latent_size=128 seed=0/"
)

an_TT = Analysis_TT(run_name="TT", filepath=pathTT)
an_TT.plot_scree(max_pcs=50)


path_GRU_Sweep = pathTT + "20240412_Fig1_MultiTask_GRU_DT_Seeds/"
subfolders_GRU = [f.path for f in os.scandir(path_GRU_Sweep) if f.is_dir()][0]

path_Vanilla_Sweep = pathTT + "20240416_Fig1_MultiTask_Vanilla_DT/"
subfolders_Vanilla = [f.path for f in os.scandir(path_Vanilla_Sweep) if f.is_dir()][0]

path_NODE_Sweep = pathTT + "20240412_Fig1_MultiTask_NODE_DT_Seeds/"
subfolders_NODE = [f.path for f in os.scandir(path_NODE_Sweep) if f.is_dir()][0]

path_LFADS_Sweep = pathTT + "20240416_Fig1_MultiTask_LFADS_DT/"
subfolders_LFADS = [f.path for f in os.scandir(path_LFADS_Sweep) if f.is_dir()][0]

analysis_GRU = Analysis_DT(
    run_name="GRU", filepath=subfolders_GRU + "/", model_type="SAE"
)
analysis_Vanilla = Analysis_DT(
    run_name="Vanilla", filepath=subfolders_Vanilla + "/", model_type="SAE"
)
analysis_NODE = Analysis_DT(
    run_name="NODE", filepath=subfolders_NODE + "/", model_type="SAE"
)
analysis_LFADS = Analysis_DT(
    run_name="LFADS", filepath=subfolders_LFADS + "/", model_type="LFADS"
)


# %%
# Plot the rates for the GRU model for a few trials
analysis_LFADS.plot_rates(phase="train", neurons=[1, 2, 3, 4, 5])
# %%
comparison = Comparison()
comparison.load_analysis(an_TT, reference_analysis=True)
comparison.load_analysis(analysis_GRU, group="GRU")
comparison.load_analysis(analysis_NODE, group="NODE")
comparison.load_analysis(analysis_Vanilla, group="Vanilla")
comparison.load_analysis(analysis_LFADS, group="LFADS")

# %%
comparison.compare_rate_r2()
# %%
rate_state = comparison.compare_state_rate_r2()
# %%
print(rate_state)
