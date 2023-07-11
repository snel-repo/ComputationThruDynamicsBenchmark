import pickle
class DataTrainWrapper():
    def __init__(self, model, data_mod, trainer):
        self.model = model
        self.trainer = trainer
        self.datamodule = data_mod
        self.seed = 0
        
    def tune_model(self):
        self.trainer.tune(model = self.model, datamodule = self.datamodule)

    def train_model(self):
        self.trainer.fit(model = self.model, datamodule = self.datamodule)

    def save_wrapper(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_wrapper(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)