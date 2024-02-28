from interpretability.comparison.analysis.tt.tt import Analysis_TT

# Create the analysis object:
analysis_tt = Analysis_TT(
    run_name="tt_3BFF_pretrained",
    filepath="/home/csverst/Github/InterpretabilityBenchmark/examples/pretrained/tt/",
)

analysis_tt.print_hyperparams()
analysis_tt.plot_trial(num_trials=2, scatterPlot=False)
