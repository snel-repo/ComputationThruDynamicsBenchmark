import os

import numpy as np
from matplotlib import pyplot as plt

from interpretability.comparison.analysis.dt.dt import Analysis_DT
from interpretability.comparison.comparison import Comparison

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HOME_DIR = "/home/csverst/Github/InterpretabilityBenchmark"
fpath_3bff_LFADS_2_DT = (
    HOME_DIR + "/content/trained_models/task-trained/20240229_3BFF_GRU_Tutorial/"
    "n=3 weight_decay=1e-08 max_epochs=1500 seed=2/20240301_NBFF_GRU_DT/"
)

dt_analysis = Analysis_DT(
    run_name="temp",
    filepath=fpath_3bff_LFADS_2_DT,
    model_type="SAE",
)

comp = Comparison()
comp.load_analysis(dt_analysis)
comp.compare_rate_r2()
rates = dt_analysis.get_rates()
true_rates = dt_analysis.get_true_rates()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.exp(rates[0, :, 0].detach().numpy()))
ax.plot(true_rates[0, :, 0].detach().numpy())
plt.savefig("test.png")
print(rates.shape, true_rates.shape)
