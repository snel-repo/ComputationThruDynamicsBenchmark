from interpretability.task_modeling.task_trained_wrapper.tasktrain_wrapper import TaskTrainWrapper
from interpretability.data_modeling.lfads_torch.data_trained_wrapper.data_trained_wrapper import DataTrainWrapper
from interpretability.comparison.comparison_wrapper import ComparisonWrapper

tt_path = "/home/csverst/Github/InterpretabilityBenchmark/task_trained_models/20230711_TBFF_RNN_Final_latent_size=64_seed=0.pkl"
dt_path = "/home/csverst/Github/InterpretabilityBenchmark/model_saves/data_trained/0711_TBFF_InputInf_RNN_Reconfig2.pkl"

cw = ComparisonWrapper(tt_path, dt_path)