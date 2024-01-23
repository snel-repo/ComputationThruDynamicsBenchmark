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
```


You will need to pip install DSA: MotorNet should be installed automatically.
See those repos for detailed instructions.
DSA: https://github.com/mitchellostrow/DSA
MotorNet: https://www.motornet.org/index.html

## Usage
The only folder needed to get a basic idea of how the package works is the scripts folder.
The two primary run scripts are "run_task_training.py" and "run_data_training.py", which train a model to perform a task, and train a model on simulated neural data from a task, respectively.

Each uses ray, hydra, and PyTorch Lightning to handle hyperparameter sweeps and logging. WandB is used by default, but TensorBoard logging is also available.

There are three primary tasks implemented, ranging from simple to complex:
1. N-bit Flip Flop (NBFF): An extension of the 3-bit Flip-Flop from OTBB, this can be extended into higher dimensions for more complex dynamics.
2. MultiTask: A version of the task used in recent papers by Yang and Driscoll, this task combines 15 simple cognitive tasks into a single task to look at how dynamical motifs can generalize across tasks.
3. MotorNet: A musculoskeletal modeling and control engine that we use to simulate a delayed RandomTarget reaching task (Codol et al.)

## Quick-Start:
To get an overview of the major components of the code-base, you should only need to run three scripts:
1. examples/run_task_training.py
2. examples/run_data_training.py
3. examples/compare_tt_dt_models.py

Before running these scripts, you will need to modify the RUNS_HOME and SAVE_PATH variables in run_task_training.py to a location where you'd like to save your training logs/plots and the fully trained final models, respectively. In addition to the simulated spiking activity, the task-training module will save a copy of the trained model, the datamodule used to train, and the simulator that generated the spiking activity.

Once the task-trained model has been run, it should save an h5 file of spiking activity in the data-trained folder. Running the data-trained model should be straightforward as well!

Once both have been run and the trained models saved to pickle files, the "compare_tt_dt_models.py" file performs basic visualizations and latent activity comparisons.


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
