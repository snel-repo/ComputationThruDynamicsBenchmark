import numpy as np
import torch
from torch.utils.data import BatchSampler


class GroupedSampler(BatchSampler):
    def __init__(self, data_source, num_samples):
        self.dataset = data_source
        self.batch_size = num_samples
        self.num_samples = len(data_source)
        self.grouped_indices = self._group_indices_by_trial_type()

    def _group_indices_by_trial_type(self):
        # Group indices by trial type.
        trial_type_indices = {}
        trial_inputs = torch.round(self.dataset.tensors[1][:, 0, 5:])
        trial_inputs = torch.nonzero(trial_inputs)[:, -1].numpy()
        for i in range(len(self.dataset)):
            trial_type = trial_inputs[i]
            if trial_type not in trial_type_indices:
                trial_type_indices[trial_type] = []
            trial_type_indices[trial_type].append(i)
        return trial_type_indices

    def __iter__(self):
        group_indices = list(self.grouped_indices.values())
        np.random.shuffle(group_indices)  # Shuffle the groups
        for group in group_indices:
            np.random.shuffle(group)  # Shuffle indices within each group
            # Yield indices batch_size at a time
            for i in range(0, len(group), self.batch_size):
                yield group[i : i + self.batch_size]

    def __len__(self):
        # Calculate batches
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class RandomSampler(BatchSampler):
    def __init__(self, data_source, num_samples):
        self.dataset = data_source
        self.batch_size = num_samples
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        # Calculate the number of batches
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class SequentialSampler(BatchSampler):
    def __init__(self, data_source, num_samples):
        self.dataset = data_source
        self.batch_size = num_samples
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = np.arange(self.num_samples)

        for i in range(0, self.num_samples, self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        # Calculate the number of batches
        return (self.num_samples + self.batch_size - 1) // self.batch_size
