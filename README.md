# Computation-Through-Dynamics Benchmark

## Overview
This git repo contains code that will allow users to perform four basic steps:
1. Train task-trained models on a range of tasks with different complexities
2. Simulate synthetic neural spiking from those task-trained networks
3. Train data-trained models on the synthetic spiking activity
4. Compare the dynamics of the task-trained and data-trained networks with a variety of quantifications of dynamical accuracy

## Installation
We recommend using Conda to run this code. Unfortunately, Ray support for Windows is spotty, so I recommend Linux.
To create an environment and install the dependencies of the project, run the following commands:

```
git clone https://github.com/snel-repo/ComputationThruDynamicsBenchmark.git
conda create --name CtDEnv python=3.10
conda activate CtDEnv
cd ComputationThruDynamicsBenchmark
pip install -e .
pip install -e libs/DSA/.
pip install -e libs/lfads-jslds/.

# To get GPUs working with jax, you have to use the CUDA 11 version:
# Here are the commands to get it working for our linux servers; find the correct version for your system.
pip uninstall jax jaxlib
conda install cuda -c nvidia/label/cuda-11.8.0
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.4+cuda11.cudnn82-cp310-cp310-manylinux2014_x86_64.whl
pip install --upgrade jax==0.4.4
conda install -c conda-forge cudnn=8.2.0.53
```

lfads-jslds is a JAX model that implements Jacobian-Switching Linear Dynamical Systems, provided by David Zoltowski.

We use MotorNet, a musculoskeletal modeling package called MotorNet from Oli Codol.
For more information on MotorNet, see the documentation:
MotorNet: https://www.motornet.org/index.html

## Usage
The only folder needed to get a basic idea of how the package works is the scripts folder.
The two primary run scripts are "run_task_training.py" and "run_data_training.py", which train a model to perform a task, and train a model on simulated neural data from a task, respectively.

Each uses ray, hydra, and PyTorch Lightning to handle hyperparameter sweeps and logging. WandB is used by default, but TensorBoard logging is also available.

There are three tasks implemented, ranging from simple to complex:
1. NBFF: An extension of the 3-bit Flip-Flop from OTBB, this can be extended into higher dimensions for more complex dynamics.
2. MultiTask: A version of the task used in recent papers by Yang and Driscoll, this task combines 15 simple cognitive tasks into a single task to look at how dynamical motifs can generalize across tasks.
3. RandomTargetDelay: A musculoskeletal modeling and control engine (MotorNet) that we use to simulate a delayed RandomTarget reaching task (Codol et al.)

## Quick-Start:
To get an overview of the major components of the code-base, you should only need to run three scripts:
1. examples/run_task_training.py
2. examples/run_data_training.py
3. examples/compare_tt_dt_models.py

Before running these scripts, you will need to modify the HOME_DIR variable in your .env file to a location where you'd like to save the outputs of the runs (datasets, logging info, trained models).

Once run_task_training.py is finished training, it will save a simulated spiking dataset in HOME_DIR/content/dataset/tt/ Modify "prefix" in run_data_training to whatever folder name is saved, typically in the form "yyyyMMdd_RUN_DESC..." Only the yyyyMMdd_RUN_DESC should be included in the prefix.
If there is more than one simulated dataset (i.e., if you did a hyperparameter sweep of task-trained models), the code currently just takes the first folder in the directory unless you pass in a "file_index" parameter into the datamodule to select a different simulated dataset.

Once run_data_training.py is complete, it will save a trained model and the datamodule as .pkl files. These pickle files will be loaded into analysis objects that have automated functions to compare models, perform fixed-point analyses, etc.

After both task-trained and data-trained models have been run, modify the dt_path and tt_path in compare_tt_dt_models.py to plot some basic comparisons and fixed-point analyses on the trained models!

## Overview of major components:
### Task-Training:
To see what tasks can specifically be implemented, look in the config files for the task trained networks. Each task is a "task_env" object, which specifies the default parameters for that task. These parameters can be modified by changing the "SEARCH_SPACE" variable in run_task_training.

#### Components of task-training pipeline:
1. callbacks: Specific routines to be run during training: Model visualizations, plotting latents, charting performance etc.
2. datamodule: Shared between tasks, handles generating data and making training/validation dataloaders
3. model: The class of model to be trained to perform the task. NODEs and RNNs have been implemented so far, but see the configs/models/ for a full list
4. simulator: The object that simulates neural activity from the task-trained network. Noise, sampling and spiking parameters can be changed here.
5. task_env: Task logic and data generation pipelines for each task.
6. task_wrapper: The class that collects all of the required components above, performs training and validation loops, configures optimizers etc.

The task-training pipeline actually generates a "train" task_env  / datamodule  and a "sim" task_env / datamodule.
The "train" versions are what is being used to train the task-trained models, while the "sim" is what is used to generate the simulated neural activity. This allows users to specify different conditions for the training and simulation pipelines, and to do more complex analyses like testing for generalization performance across task types.

### Simulation:
The simulator's instance variables contains the parameters for the neural data simulation. There are options to change the noise model for the simulation, change the number of simulated neurons, and whether to embed the latent activity onto a non-linear manifold prior to sampling spiking activity (experimental).

The main method for this object is "simulate_neural_data", which takes in a trained model, a datamodule with the trials to simulate neural activity from, the run tag, path variables, and a random seed. This method saves an h5 file of spiking activity (along with other variables that might be needed for training, e.g., inputs etc.) in the "content/datasets/dt/" folder.

### Data-Training:
Runs with either a generic SAE or LFADS models (currently). Whether to use a generic SAE or LFADS is controlled by the MODEL_CLASS variable, which for now is either SAE or LFADS.

### Comparisons:
Comparator object takes in Analysis objects with specific return structures.
Comparator is agnostic to the origin of the dataset, can operate equivalently on task-trained and data-trained models.

## Contributing
Talk to me!

## License
None yet

There are three major items that are saved by these scripts, following this directory structure:
- runs: The logs of the training runs produced by WandB.
- datasets: .h5 and .pkl files of the datasets used at different phases of the pipeline:
- - tt: Data for training the task-trained model
- - sim: Data passed through the task-trained model to generate simulated spiking activity
- - dt: Simulated spiking activity for data-trained models
- trained_models: Where the datamodule and trained models are saved for use by Analysis objects.
- - task-trained:
- - - datamodule_train.pkl: Pickled PyTorch Lightning datamodule used to train model
- - - datamodule_sim.pkl: Pickled PyTorch Lightning datamodule used to simulate neural activity
- - - model.pkl: Trained PyTorchLightning module
- - - simulator.pkl: Object used to simulate neural spiking activity
- - data-trained:
- - - datamodule.pkl: Pickled datamodule object for data-training
- - - model.pkl: Trained model object


## Contact
chrissversteeg@gmail.com for questions/concerns!

## Acknowledgments
Thanks to a lot of people, will populate before release
