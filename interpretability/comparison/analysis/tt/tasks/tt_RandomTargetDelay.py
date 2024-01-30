import matplotlib.pyplot as plt

from interpretability.comparison.analysis.tt.tt import Analysis_TT


class TT_RandomTargetDelay(Analysis_TT):
    def __init__(self, run_name, filepath):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath)
        self.plot_path = (
            "/home/csverst/Github/InterpretabilityBenchmark/"
            f"interpretability/comparison/plots/{self.run_name}/"
        )

    def plotTrial(self, trial_num):
        # plot the trial
        # get the trial
        tt_ics, tt_inputs, tt_targets = self.get_model_input()
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        controlled = out_dict["controlled"]

        targets_trial = tt_targets[trial_num, :, :].detach().numpy()
        controlled_trial = controlled[trial_num, :, :].detach().numpy()

        # plot the trial
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Put a square at the start location (targets_trial)
        ax.plot(
            targets_trial[0, 0],
            targets_trial[0, 1],
            marker="s",
            color="r",
            markersize=10,
        )
        ax.plot(
            targets_trial[-1, 0],
            targets_trial[-1, 1],
            marker="s",
            color="g",
            markersize=8,
        )

        ax.plot(
            controlled_trial[:, 0],
            controlled_trial[:, 1],
            color="k",
        )
