from interpretability.comparison.analysis.analysis_tt import Analysis_DT

top_dir = "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
filepaths = [
    "20231130_NBFF_GRU/",
    "20231130_NBFF_NODE_higherLR/",
    "20231201_NBFF_GRU_4Bit/",
    "20231201_NBFF_NODE_4Bit/",
    "20231201_RSG_GRU_64D/",
    "20231201_RSG_NODE_5D/",
]

tbff = Analysis_DT(run_name="TBFF", filepath=top_dir + filepaths[0])
x = 1
