import h5py
import torch

from ctd.comparison.analysis.analysis import Analysis


class Analysis_Ext(Analysis):
    def __init__(self, run_name, filepath):
        self.run_name = run_name
        self.filepath = filepath

        self.latents = None
        self.rates = None
        self.load_data(filepath)

    def load_data(self, filepath):
        with h5py.File(filepath, "r") as h5file:
            # Check the fields
            print(h5file.keys())
            self.latents = torch.Tensor(h5file["jslds_factors"][()])
            self.rates = torch.Tensor(h5file["jslds_rates"][()])

    def get_latents(self):
        return self.latents

    def get_rates(self):
        return self.rates

    def get_model_output(self):
        return self.latents, self.rates

    def compute_FPs(self, latents, inputs):
        return None
