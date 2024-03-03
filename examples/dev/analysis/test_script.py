import os
import pickle

from interpretability.data_modeling.datamodules.LFADS.datamodule import BasicDataModule

SAVE_PATH = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "pretrained/dt/20240302_NBFF_LFADS_DT/"
)


path3 = os.path.join(SAVE_PATH, "model.pkl")
# model = pickle.load(open(path3, "rb"))
# path4 = os.path.join(SAVE_PATH, "datamodule_gpu.pkl")
# datamodule = pickle.load(open(path4, "rb"))

# path5 = os.path.join(SAVE_PATH, "model_cpu.pkl")
dm = BasicDataModule(
    prefix="20240229_3BFF_GRU_Tutorial",
    system="3BFF",
    gen_model="GRU_RNN",
    n_neurons=50,
    nonlin_embed=True,
    obs_noise="poisson",
    seed=0,
)

model = pickle.load(open(path3, "rb"))
dm.prepare_data()
dm.setup()
model = model.to("cpu")
path6 = os.path.join(SAVE_PATH, "datamodule.pkl")
# Save the datamodule
with open(path6, "wb") as f:
    pickle.dump(dm, f)

# Save the model
path7 = os.path.join(SAVE_PATH, "model.pkl")
with open(path7, "wb") as f:
    pickle.dump(model, f)
