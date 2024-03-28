import os
import pickle
from pathlib import Path

import dotenv

dotenv.load_dotenv()
HOME_DIR = Path(os.environ.get("HOME_DIR"))

path = (
    HOME_DIR
    / "pretrained"
    / "20240327_RandomTarget_GRU_Final"
    / "max_epochs=1500 seed=0"
)
fields_to_remove = [
    "train_ds",
    "valid_ds",
    "all_data",
]

for file in path.glob("**/datamodule_sim.pkl"):
    # load the datamodules
    with open(file, "rb") as f:
        file1 = pickle.load(f)
        # Remove the presaved data
        for field in fields_to_remove:
            if hasattr(file1, field):
                delattr(file1, field)

    # Save the datamodule back
    with open(file, "wb") as f:
        pickle.dump(file1, f)

        x = 1
