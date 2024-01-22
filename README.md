# Computation-Through-Dynamics Benchmark

## Overview
This git repo contains code that will allow users to perform four basic steps:
1. Train task-trained models on a range of tasks with different complexities
2. Simulate synthetic neural spiking from those task-trained networks
3. Train data-trained models on the synthetic spiking activity
4. Compare the dynamics of the task-trained and data-trained networks with a variety of quantifications of dynamical accuracy

## Installation
We recommend using Conda to run this code.
To install dependencies, you can use the interpretabilityEnv.yaml file

conda env create -f environment.yaml
conda activate CtDEnv

You may need to pip install DSA and MotorNet manually: see those repos for detailed instructions.

## Usage
At a high level, the only folder that a user will need to understand is the scripts folder.
The two primary run scripts are "run_task_training.py" and "run_data_training.py", which do what the label says.
Each uses ray, hydra, and PyTorch Lightning to handle hyperparameter sweeps and logging. WandB is used by default, but TensorBoard logging is also available.

There are three primary tasks implemented, ranging from simple to complex:
1. N-bit Flip Flop (NBFF): An extension of the 3-bit Flip-Flop from OTBB, this can be extended into higher dimensions for more complex dynamics.
2. MultiTask: A version of the task used in recent papers by Yang and Driscoll, this task combines 15 simple cognitive tasks into a single task to look at how dynamical motifs can generalize across tasks.
3. MotorNet: A musculoskeletal modeling and control engine that we use to simulate a delayed RandomTarget reaching task (Codol et al.)

## Quick-Start:
To get an overview of the major components of the code-base, you should only need to run three scripts:
1. scripts/run_task_training
2. scripts/run_data_training
3. scripts/compare_datasets

### Task-Training:
To see what tasks can specifically be implemented, look in the config files for the task trained networks. Each task is a "task_env" object, which specifies the default parameters for that task. These parameters can be modified by changing the "SEARCH_SPACE" variable in run_task_training.

#### Components of task-training pipeline:
1. callbacks: Specific routines to be run during training: Model visualizations, plotting latents, charting performance etc.
2. datamodule: Shared between tasks, handles generating data and making training/validation dataloaders
3. model: The class of model to be trained to perform the task. NODEs and RNNs have been implemented so far, but see the configs/models/ for a full list
4. simulator: The object that simulates neural activity from the task-trained network. Noise, sampling and spiking parameters can be changed here.
5. task_env: Task logic and data generation pipelines for each task.
6. task_wrapper: The class that collects all of the required components above, performs training and validation loops, configures optimizers etc.

### Simulation:
TODO

### Data-Training:
TODO

### Comparisons:
MOST TODO, but as I am envisioning it, it will be a Comparator object which will load in TT and DT models, then perform comparisons like DSA, FP analysis, etc. within it, rather than relying on external scripts.


## Contributing
Talk to me!

## License
None yet

## Contact
chrissversteeg@gmail.com for questions/concerns!

## Acknowledgments
Thanks to a lot of people, will populate before release
