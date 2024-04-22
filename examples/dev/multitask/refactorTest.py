import numpy as np
import torch

from ctd.comparison.analysis.tt.tasks.tt_MultiTask import Analysis_TT_MultiTask

# suffix = "MultiTaskGRU_LowLR"
# filepath1 = (
# "/home/csverst/Github/InterpretabilityBenchmark/"
# "trained_models/task-trained/20240116_MultiTask_GRU_lowLR/"
# )

suffix = "MultiTaskGRU_Final3"
filepath1 = (
    "/home/csverst/Github/CtDBenchmark/content/trained_models/"
    + "task-trained/20240418_MultiTask_NoisyGRU_Final/"
    + "max_epochs=500 latent_size=64 seed=0/"
)

comp = Analysis_TT_MultiTask(run_name=suffix, filepath=filepath1)

flag1, phase_dict = comp.get_task_flag(task_to_analyze="MemoryPro")

extra = comp.datamodule.extra_data

train_ds = comp.datamodule.train_ds
valid_ds = comp.datamodule.valid_ds

train_ics = train_ds.tensors[0]
train_inputs = train_ds.tensors[1]
train_targets = train_ds.tensors[2]
train_inds = train_ds.tensors[3].detach().numpy().astype(int)
train_conds = train_ds.tensors[4]
train_true_inputs = train_ds.tensors[7]

valid_ics = valid_ds.tensors[0]
valid_inputs = valid_ds.tensors[1]
valid_targets = valid_ds.tensors[2]
valid_inds = valid_ds.tensors[3].detach().numpy().astype(int)
valid_conds = valid_ds.tensors[4]
valid_true_inputs = valid_ds.tensors[7]

ics = torch.cat([train_ics, valid_ics], dim=0)
inputs = torch.cat([train_inputs, valid_inputs], dim=0)
targets = torch.cat([train_targets, valid_targets], dim=0)
inds = np.concatenate([train_inds, valid_inds], axis=0)
conds = torch.cat([train_conds, valid_conds], dim=0)
true_inputs = torch.cat([train_true_inputs, valid_true_inputs], dim=0)


phase_dict_train = phase_dict_train = [phase_dict[ind] for ind in train_inds]
phase_dict_valid = phase_dict_valid = [phase_dict[ind] for ind in valid_inds]
phase_dict = phase_dict_train + phase_dict_valid
