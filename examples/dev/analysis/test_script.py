import io
import os
import pickle

import torch

from interpretability.comparison.analysis.dt.dt import Analysis_DT


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


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
