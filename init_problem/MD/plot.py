import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .potentials import compute_potential

#DATA = "train_data/3D_double_well_adjusted_1e8steps_0e0burn_20261030_1446.npz"
DATA='/Users/clarawimmelmann/Desktop/Energy_specialCourse/init_problem/MD/train_data/3D_double_well_adjusted_1e8steps_0e0burn_20261102_0733.npz'
MAX_PLOT_POINTS = 100_000


def count_well_transitions(positions, potential_type, period=32.0):
    """Count sampled changes between wells across all trajectories."""
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[2] < 1:
        raise ValueError("positions must have shape (trajectories, frames, dimensions)")

    x = positions[..., 0]
    if potential_type == "chain":
        centers = (period / 8 * np.arange(8)[:, None] + [-1.0, 1.0]).ravel()
        distances = np.abs(
            (x[..., None] - centers + period / 2) % period - period / 2
        )
        well_indices = np.argmin(distances, axis=-1)
    else:
        well_indices = x >= 0
    return int(np.count_nonzero(np.diff(well_indices, axis=1)))


def plot_simulation(
    data_path: Path,
    output_plot_path: Path = None,
    trajectory_ids: list[int] | None = None,
    stop_time: float | None = None,
):
    with np.load(data_path) as archive:
        positions = np.asarray(archive["positions"])
        all_energies = np.asarray(archive["energy"])
        potential_type = str(archive["potential_type"].item())
        well_amplitude = float(archive["well_amplitude"])
        md_time = (
            np.asarray(archive["md_time"], dtype=float)
            if "md_time" in archive
            else np.arange(positions.shape[1], dtype=float)
        )

    if positions.ndim != 3 or positions.shape[2] < 2:
        raise ValueError("positions must have shape (trajectories, frames, at least 2)")
    if md_time.ndim != 1 or len(md_time) != positions.shape[1]:
        raise ValueError("md_time must match the saved position frames")
    if not np.isfinite(md_time).all() or np.any(np.diff(md_time) < 0):
        raise ValueError("md_time must contain finite, ordered times")
    if stop_time is None:
        stop_time = float(md_time[-1])
    if not np.isfinite(stop_time) or stop_time < md_time[0] or stop_time > md_time[-1]:
        raise ValueError(
            f"stop_time must be between {md_time[0]:g} and {md_time[-1]:g}"
        )
    if all_energies.shape != positions.shape[:2]:
        raise ValueError("energy must match the trajectory and frame dimensions")

    total_trajectories = positions.shape[0]
    if trajectory_ids is None:
        selected_indices = np.arange(total_trajectories)
    else:
        if not trajectory_ids:
            raise ValueError("trajectory_ids must contain at least one trajectory ID")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("trajectory_ids must not contain duplicates")
        invalid_ids = [
            trajectory_id
            for trajectory_id in trajectory_ids
            if trajectory_id < 1 or trajectory_id > total_trajectories
        ]
        if invalid_ids:
            raise ValueError(
                f"Trajectory IDs must be between 1 and {total_trajectories}; "
                f"got {invalid_ids}"
            )
        selected_indices = np.asarray(trajectory_ids) - 1

    selected_ids = selected_indices + 1
    n_trajectories = len(selected_indices)
    spatial_positions = positions[selected_indices, ..., :2]
    selected_energies = all_energies[selected_indices]
    all_spatial_positions = positions[..., :2]
    x_flat = spatial_positions[..., 0].flatten()
    y_flat = spatial_positions[..., 1].flatten()

    box_length = 32.0 if potential_type == "chain" else None
    x_min, x_max = (0.0, box_length) if potential_type == "chain" else (-2.5, 2.5)

    fig = plt.figure(figsize=(16, 17))
    gs = fig.add_gridspec(3, 2)

    # --- All-sample density over the potential ---
    ax1 = fig.add_subplot(gs[0, 0])
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_pot = compute_potential(
        X, Y, potential_type, period=box_length or 32.0,
        well_amplitude=well_amplitude,
    )
    contour = ax1.contour(X, Y, Z_pot, levels=12, cmap="Greys", alpha=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)
    density_x_edges = np.linspace(x_min, x_max, 101)
    density_y_edges = np.linspace(-2.0, 2.0, 81)
    sample_counts = np.zeros((100, 80), dtype=np.int64)
    for trajectory in all_spatial_positions:
        counts, _, _ = np.histogram2d(
            trajectory[:, 0],
            trajectory[:, 1],
            bins=(density_x_edges, density_y_edges),
        )
        sample_counts += counts.astype(np.int64)
    density = ax1.pcolormesh(
        density_x_edges,
        density_y_edges,
        sample_counts.T,
        cmap="plasma",
        shading="auto",
    )
    fig.colorbar(density, ax=ax1, label="Sample count")
    ax1.set_title("All MD Samples over $V(x,y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # --- Selected trajectories through the potential ---
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contour(X, Y, Z_pot, levels=12, cmap="Greys", alpha=0.5)
    ax2.clabel(contour, inline=True, fontsize=8)
    trajectory_stride = max(
        1, (spatial_positions.shape[1] + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS
    )
    if n_trajectories > 1 or trajectory_ids is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        for traj_idx in range(n_trajectories):
            ax2.plot(
                spatial_positions[traj_idx, ::trajectory_stride, 0],
                spatial_positions[traj_idx, ::trajectory_stride, 1],
                color=colors[traj_idx],
                alpha=0.7,
                linewidth=0.8,
                label=f"Traj {selected_ids[traj_idx]}"
                if n_trajectories <= 10
                else "",
            )
        ax2.set_title(
            f"2D Paths for Trajectories {', '.join(map(str, selected_ids))} over $V(x,y)$"
        )
        if n_trajectories <= 10:
            ax2.legend(loc="upper right", fontsize=7)
    else:
        ax2.plot(
            spatial_positions[0, ::trajectory_stride, 0],
            spatial_positions[0, ::trajectory_stride, 1],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.7,
            label=f"Traj {selected_ids[0]}",
        )
        ax2.set_title(f"2D Path for Trajectory {selected_ids[0]} over $V(x,y)$")
        ax2.legend(loc="upper right", fontsize=7)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    # --- Energy Distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(selected_energies.flatten(), bins=40, density=True, color="#6ba2c6", edgecolor="black", alpha=0.75)
    ax3.set_title("Boltzmann Energy Distribution")
    ax3.set_xlabel("Potential Energy $U(x, y)$")
    ax3.set_ylabel("Probability Density")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # ---  Marginal Distribution $p(x)$ & Barrier ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4_twin = ax4.twinx()
    ax4.hist(x_flat, bins=60, density=True, color="teal", alpha=0.6, label="Sample Density $p(x)$")
    x_line = np.linspace(x_min, x_max, 300)
    v_slice = compute_potential(
        x_line, np.zeros_like(x_line), potential_type, period=box_length or 32.0,
        well_amplitude=well_amplitude,
    )
    ax4_twin.plot(x_line, v_slice, color="black", lw=2, linestyle="--", label="$V(x, 0)$")
    ax4.set_title("Marginal Distribution $p(x)$ & Energy Barrier")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Probability Density", color="teal")
    ax4_twin.set_ylabel("Potential Energy $V(x, 0)$", color="black")

    # --- Initial saved interval, from burn-in through the requested time ---
    ax5 = fig.add_subplot(gs[2, 0])
    x_traj_single = spatial_positions[0, :, 0]
    y_traj_single = spatial_positions[0, :, 1]
    initial_time_mask = md_time <= stop_time
    initial_times = md_time[initial_time_mask]
    if not len(initial_times):
        raise ValueError("stop_time excludes all saved MD frames")
    initial_x = x_traj_single[initial_time_mask]
    initial_y = y_traj_single[initial_time_mask]
    time_series_stride = max(
        1, (len(initial_times) + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS
    )

    ax5.plot(
        initial_times[::time_series_stride],
        initial_x[::time_series_stride],
        label="x(t)",
        color="indigo",
        alpha=0.8,
        lw=0.8,
    )
    ax5.plot(
        initial_times[::time_series_stride],
        initial_y[::time_series_stride],
        label="y(t)",
        color="mediumseagreen",
        alpha=0.6,
        lw=0.8,
    )
    ax5.set_title(
        f"Trajectory {selected_ids[0]} from burn-in "
        f"(t={md_time[0]:g} to {stop_time:g})"
    )
    ax5.set_xlabel("MD time")
    ax5.set_ylabel("Position Coordinates")
    ax5.legend(loc="upper right", fontsize=8)

    # --- Full saved trajectory, with total MD transition count ---
    ax6 = fig.add_subplot(gs[2, 1])
    time_series_stride = max(
        1, (len(md_time) + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS
    )
    ax6.plot(
        md_time[::time_series_stride],
        x_traj_single[::time_series_stride],
        label="x(t)",
        color="indigo",
        alpha=0.8,
        lw=0.8,
    )
    ax6.plot(
        md_time[::time_series_stride],
        y_traj_single[::time_series_stride],
        label="y(t)",
        color="mediumseagreen",
        alpha=0.6,
        lw=0.8,
    )
    transition_count = count_well_transitions(
        positions[selected_indices[0]:selected_indices[0] + 1],
        potential_type,
        period=box_length or 32.0,
    )
    ax6.set_title(
        f"Trajectory {selected_ids[0]} over all saved MD times"
    )
    ax6.set_xlabel("MD time")
    ax6.set_ylabel("Position Coordinates")
    ax6.legend(
        loc="upper right",
        fontsize=8,
        title=f"Trajectory {selected_ids[0]} well transitions: {transition_count}",
    )

    plt.tight_layout()
    if output_plot_path is None:
        output_plot_path = Path(data_path).with_suffix(".png")
    plt.savefig(output_plot_path, dpi=300)
    plt.show()
    print(f"Saved visualization to {output_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MD simulation archive")
    parser.add_argument("--data", type=Path, default=Path(DATA), help="Path to .npz simulation file")
    parser.add_argument(
        "--trajectories",
        type=int,
        nargs="+",
        metavar="ID",
        help="1-based trajectory IDs to plot (default: all)",
    )
    parser.add_argument(
        "--stop-time",
        type=float,
        help="End physical MD time for the initial trajectory panel (default: end of archive)",
    )
    args = parser.parse_args()
    plot_simulation(
        args.data,
        trajectory_ids=args.trajectories,
        stop_time=args.stop_time,
    )
