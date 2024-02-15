import numpy as np
from torch.utils.data import BatchSampler


class GroupedSampler(BatchSampler):
    # This sampler yields batches of data grouped by trial type.
    # This is useful for getting shared motifs on the MultiTask dataset
    def __init__(self, data_source, num_samples):
        self.dataset = data_source
        self.batch_size = num_samples
        self.num_samples = len(data_source)
        self.grouped_indices = self._group_indices_by_trial_type()

    def _group_indices_by_trial_type(self):
        # Group indices by trial type.
        trial_type_indices = {}
        trial_type = self.dataset.tensors[4]
        unique_trial_types = np.unique(trial_type)
        for ind1, trial_type1 in enumerate(unique_trial_types):
            trial_type_indices[ind1] = np.where(trial_type == trial_type1)[0]
        return trial_type_indices

    def __iter__(self):
        group_indices = list(self.grouped_indices.values())
        indices_lens = np.array([len(x) for x in group_indices])
        group_counter = np.zeros(len(group_indices)).astype(int)
        np.random.shuffle(group_indices)  # Shuffle the groups
        while np.any(group_counter < indices_lens):
            for i, group in enumerate(group_indices):
                if group_counter[i] < len(group):
                    yield group[group_counter[i] : group_counter[i] + self.batch_size]
                    group_counter[i] += self.batch_size

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
