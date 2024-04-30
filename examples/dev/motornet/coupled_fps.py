import os

# Import pca
import dotenv

from ctd.comparison.analysis.tt.tasks.tt_RandomTarget import TT_RandomTarget

dotenv.load_dotenv(dotenv.find_dotenv())

HOME_DIR = os.environ["HOME_DIR"]
print(HOME_DIR)

pathTT = (
    HOME_DIR
    + "content/trained_models/task-trained/20240419_RandomTarget_"
    + "NoisyGRU_Final/max_epochs=1500 latent_size=64 seed=0/"
)
an_TT = TT_RandomTarget(run_name="TT", filepath=pathTT)

tt_fps = an_TT.compute_coupled_FPs(
    learning_rate=1e10,
    noise_scale=0.0,
    n_inits=2000,
    max_iters=10000,
    device="cpu",
)
