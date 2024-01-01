import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

PLOT_PATH = (
    "/home/csverst/Github/InterpretabilityBenchmark/interpretability/comparison/plots/"
)


def animate_latent_trajectories(
    latents,
    x_dict,
    y_dict,
    z_dict,
    align_idx,
    align_str,
    pre_align,
    post_align,
    trial_coloring,
    trail_length=2,
    filename="aligned_latent_trajectory.mp4",
    elev=None,
    azim=None,
    plot_title="",
    suffix="",
):
    # Step 1: Define the 3D subspace and colormap
    x_vec = x_dict["vector"]
    y_vec = y_dict["vector"]
    z_vec = z_dict["vector"]

    # Normalize target positions and create a colormap
    norm = mcolors.Normalize(vmin=trial_coloring.min(), vmax=trial_coloring.max())
    colormap = cm.hsv

    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(x_dict["label"])
    ax.set_ylabel(y_dict["label"])
    ax.set_zlabel(z_dict["label"])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_title(plot_title)
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)

    time_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes)
    # Step 2: Project aligned latents and prepare for animation
    max_frames = 0
    lines = []
    data = []
    for batch_index in range(latents.shape[0]):
        # Extract aligned snippet for each batch
        if (
            align_idx[batch_index] - pre_align < 0
            or align_idx[batch_index] + post_align > latents.shape[1] - 1
        ):
            continue
        start_idx = max(align_idx[batch_index] - pre_align, 0)
        end_idx = min(align_idx[batch_index] + post_align, latents.shape[1])
        batch_latents = latents[batch_index, start_idx:end_idx]  # Aligned snippet

        projected_latents = np.dot(
            batch_latents, np.vstack([x_vec, y_vec, z_vec]).T
        )  # Snippet x 3

        max_frames = max(max_frames, projected_latents.shape[0])
        data.append(
            [projected_latents[:, 0], projected_latents[:, 1], projected_latents[:, 2]]
        )

        # Get color for the line based on target position
        color = colormap(norm(np.linalg.norm(trial_coloring[batch_index])))
        (line,) = ax.plot([], [], [], lw=2, color=color)
        lines.append(line)

    # Step 3: Define the update function for animation
    def update(num, data, lines):
        for line, d in zip(lines, data):
            end = min(num + trail_length, len(d[0]))
            start = max(0, end - trail_length)
            line.set_data(d[0][start:end], d[1][start:end])
            line.set_3d_properties(d[2][start:end])
        current_time = num - pre_align
        time_text.set_text(f"Time rel. {align_str}: {current_time}")
        return lines

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, fargs=(data, lines), interval=100
    )

    # Step 4: Save the animation
    ani.save(PLOT_PATH + suffix + "/" + filename, writer="ffmpeg")

    plt.close(fig)  # Close the figure to prevent it from displaying in Jupyter notebook
