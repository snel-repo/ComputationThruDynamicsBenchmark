import os

from interpretability.comparison.analysis.dt.dt import Analysis_DT

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HOME_DIR = "/home/csverst/Github/InterpretabilityBenchmark"
fpath_3bff_LFADS_2_DT = (
    HOME_DIR + "/content/trained_models/task-trained/20240229_3BFF_GRU_Tutorial/"
    "n=3 weight_decay=1e-08 max_epochs=1500 seed=2/20240303_NBFF_LFADS_DT_CPU/"
)

dt_analysis = Analysis_DT(
    run_name="temp",
    filepath=fpath_3bff_LFADS_2_DT,
    model_type="LFADS",
)
dt_analysis.plot_fps(device="cpu")
