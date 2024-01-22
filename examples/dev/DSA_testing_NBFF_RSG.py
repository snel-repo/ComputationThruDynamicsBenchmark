# %%

import plotly.graph_objects as go

from interpretability.comparison.analysis.analysis_tt import MultiComparator

# plt.ion()
# %%
top_dir = "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
filepaths = [
    "20231130_NBFF_GRU/",
    "20231130_NBFF_NODE_higherLR/",
    "20231201_NBFF_GRU_4Bit/",
    "20231201_NBFF_NODE_4Bit/",
    "20231201_RSG_GRU_64D/",
    "20231201_RSG_NODE_5D/",
]

labels = ["GRU_3BFF", "NODE_3BFF", "GRU_4BFF", "NODE_4BFF", "GRU_RSG", "NODE_RSG"]


comp_multi = MultiComparator(suffix="MultipleComparisons")
for filepath in filepaths:
    comp_multi.load_task_train_wrapper(filepath=top_dir + filepath)

ranks = [70]
n_delays = [75]
d_interval = 1
num_PCs = 30
for rank in ranks:
    for n_delay in n_delays:
        similarities = comp_multi.perform_dsa(
            n_delays=n_delay,
            rank=rank,
            delay_interval=d_interval,
            verbose=False,
            iters=1000,
            lr=1e-2,
            num_PCs=num_PCs,
        )

        # Plot the results as a covariance matrix
        fig = go.Figure(
            data=go.Heatmap(
                z=similarities,
                x=labels,
                y=labels,
                hoverongaps=False,
                colorscale="Viridis"
                # log-scale the colormap
            )
        )
        fig.update_layout(
            title="Similarity Matrix",
            xaxis_title="Model 1",
            yaxis_title="Model 2",
            font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        )
        # Flip the y-axis
        fig.update_yaxes(autorange="reversed")
        # invert the colormap
        fig.update_traces(colorscale="Viridis_r")
        # Make figure square
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
            paper_bgcolor="LightSteelBlue",
        )

        fig.show()
        # Save as png
        fig.write_image(f"similarity_matrix_rank{rank}_nDelay{n_delay}.png")
        # fig.write_html("similarity_matrix.html")

# %%
