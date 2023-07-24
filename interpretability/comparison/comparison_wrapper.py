import numpy as np
from data_trained_wrapper import DataTrainWrapper

# import lr from sklearn
from sklearn.linear_model import LinearRegression

from interpretability.task_modeling.task_trained_wrapper.tasktrain_wrapper import (
    TaskTrainWrapper,
)


class ComparisonWrapper:
    def __init__(self, task_train_path, data_train_path):
        task_train = TaskTrainWrapper.load_wrapper(filepath=task_train_path)
        self.env = task_train.task_env
        self.task_model = task_train.model
        self.simulator = task_train.data_simulator
        self.task_datamod = task_train.datamodule
        data_train = DataTrainWrapper.load_wrapper(filepath=data_train_path)
        self.data_model = data_train.model
        self.data_datamod = data_train.datamodule

        self.task_model.eval()  # set to evaluation mode
        self.data_model.eval()  # set to evaluation mode

    def compare_latent_activity(self):
        valid_dataloader_TT = self.task_datamod.train_dataloader(shuffle=False)
        latents_TT = []
        for i, batch in enumerate(valid_dataloader_TT):
            _, latents_temp = self.task_model(batch[1])
            latents_TT.append(latents_temp.detach().numpy())
        latents_TT = np.concatenate(latents_TT, axis=0)

        valid_dataloader_DT = self.data_datamod.train_dataloader(shuffle=False)
        latents_DT = []
        for i, batch in enumerate(valid_dataloader_DT):
            outputs_DT = self.data_model.predict_step(batch, batch_ix=i)
            latents_DT.append(outputs_DT[0][6].detach().numpy())
        latents_DT = np.concatenate(latents_DT, axis=0)

        # Fit a linear model from latents_TT to latents_DT
        latents_TT = latents_TT.reshape(-1, latents_TT.shape[-1])
        latents_DT = latents_DT.reshape(-1, latents_DT.shape[-1])
        lr_TT2DT = LinearRegression().fit(latents_TT, latents_DT)
        lr_DT2TT = LinearRegression().fit(latents_DT, latents_TT)

        r2_TT2DT = lr_TT2DT.score(latents_TT, latents_DT)
        r2_DT2TT = lr_DT2TT.score(latents_DT, latents_TT)
        return r2_TT2DT, r2_DT2TT
