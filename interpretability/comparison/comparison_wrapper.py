import pytorch_lightning as pl
import torch
from interpretability.task_modeling.task_trained_wrapper.tasktrain_wrapper import TaskTrainWrapper
from interpretability.data_modeling.lfads_torch.data_trained_wrapper.data_trained_wrapper import DataTrainWrapper

class ComparisonWrapper:
    def __init__(self, task_train_path, data_train_path):
        task_train = TaskTrainWrapper.load_wrapper(filepath=task_train_path)
        self.env = task_train.task_env
        self.task_model = task_train.model
        self.simulator = task_train.data_simulator
        data_train = DataTrainWrapper.load_wrapper(filepath=data_train_path)
        self.data_model = data_train.model
        self.data_datamod = data_train.datamodule

        self.task_model.eval()  # set to evaluation mode
        self.data_model.eval()  # set to evaluation mode

    def compare_latent_activity(self):
        spikes, activity, latents, inputs = self.simulator.simulate_new_data(self.task_net, self.synthetic_data, seed=0)
        latents_TT = torch.from_numpy(latents)
        _, latents_DT = self.data_net(spikes)
        latents_DT = latents_DT.detach().numpy()
        return latents_TT, latents_DT

