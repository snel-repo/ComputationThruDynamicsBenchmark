from ctd.comparison.analysis.tt.tt import Analysis_TT
import dotenv
import os
import pickle
import shutil
import os

def copy_folder_contents(src_folder, dest_folder):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate over all files and directories in the source folder
    for item_name in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item_name)
        dest_item = os.path.join(dest_folder, item_name)

        # If it's a directory, copy it recursively
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dest_item)

dotenv.load_dotenv(override=True)
HOME_DIR = os.environ.get("HOME_DIR")

tt_3bff_path = HOME_DIR +  "pretrained/20240503_Fig1_NBFF_NoisyGRU/"
# tt_MultiTask_path = HOME_DIR + "pretrained/20240503_MultiTask_NoisyGRU/"
tt_RandomTarget_path = HOME_DIR + "pretrained/20240605_RandomTarget_NoisyGRU_GoStep_ModL2_Delay/"

tt_3bff = Analysis_TT(run_name = "tt_3bff", filepath = tt_3bff_path)
# tt_MultiTask = Analysis_TT(run_name = "tt_MultiTask", filepath = tt_MultiTask_path)
tt_RandomTarget = Analysis_TT(run_name = "tt_RandomTarget", filepath = tt_RandomTarget_path)

# Make copies of the pretrained models to the trained_models folder
copy_folder_contents(tt_3bff_path, HOME_DIR + "content/trained_models/task-trained/tt_3bff/")
# copy_folder_contents(tt_MultiTask_path, HOME_DIR + "content/trained_models/task-trained/tt_MultiTask/")
copy_folder_contents(tt_RandomTarget_path, HOME_DIR + "content/trained_models/task-trained/tt_RandomTarget/")

# Generate simulated datasets
dataset_path = HOME_DIR + "content/datasets/dt/"

tt_3bff.simulate_neural_data(
    subfolder = "max_epochs=500 n_samples=1000 latent_size=64 seed=0 learning_rate=0.001",
    dataset_path=dataset_path
    )

# tt_MultiTask.simulate_neural_data(
#     subfolder = "max_epochs=2000 latent_size=128 l2_wt=5e-05 proprioception_delay=0.02 vision_delay=0.05 n_samples=1100 n_samples=1100 seed=0 learning_rate=0.005", 
#     dataset_path=dataset_path
#     )


tt_RandomTarget.simulate_neural_data(
    subfolder = "max_epochs=2000 latent_size=128 l2_wt=5e-05 proprioception_delay=0.02 vision_delay=0.05 n_samples=1100 n_samples=1100 seed=0 learning_rate=0.005", 
    dataset_path=dataset_path
    )