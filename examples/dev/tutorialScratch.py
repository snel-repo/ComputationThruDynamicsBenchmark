import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn

from interpretability.task_modeling.callbacks.callbacks import StateTransitionCallback
from interpretability.task_modeling.datamodule.task_datamodule import TaskDataModule
from interpretability.task_modeling.task_env.task_env import NBitFlipFlop
from interpretability.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper


class Vanilla_Cell(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.Whh = nn.Linear(latent_size, latent_size, bias=False)
        self.Wxh = nn.Linear(input_size, latent_size, bias=False)
        self.activation = nn.Tanh()
        self.bias = nn.Parameter(torch.zeros(latent_size))

    def forward(self, inputs, hidden=None):
        hidden = self.activation(self.Whh(hidden) + self.Wxh(inputs) + self.bias)
        return hidden


class Tutorial_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = Vanilla_Cell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden=None):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


# Create the task environment:
env_3bff = NBitFlipFlop(n=3, n_timesteps=500, switch_prob=0.01, noise=0.1)
input_size = env_3bff.observation_space.shape[0] + env_3bff.context_inputs.shape[0]
output_size = env_3bff.action_space.shape[0]

# Create the task datamodule:
dm = TaskDataModule(env_3bff)

# Create the task model:
model = Tutorial_RNN(latent_size=64)
model.init_model(input_size=input_size, output_size=output_size)

# Create the task wrapper:
wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay=1e-6)
wrapper.set_environment(env_3bff)
wrapper.set_model(model)

lg = WandbLogger(project="task_modeling")
cb = StateTransitionCallback(log_every_n_epochs=1)
# Train the task model:
trainer = pl.Trainer(max_epochs=10, callbacks=[cb])

trainer.fit(wrapper, dm)
