import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .potentials import compute_potential

#DATA = "train_data/3D_double_well_adjusted_1e8steps_0e0burn_20261030_1446.npz"
DATA='/Users/clarawimmelmann/Desktop/Energy_specialCourse/init_problem/MD/train_data/3D_double_well_adjusted_1e8steps_0e0burn_20261102_0733.npz'
MAX_PLOT_POINTS = 100_000

def plot_simulation(
    data_path: Path,
    output_plot_path: Path = None,
    trajectory_ids: list[int] | None = None,
):
    archive = np.load(data_path)
    positions = archive["positions"]  # Shape: (n_trajectories, n_saved, dim)
    all_energies = archive["energy"]
    potential_type = str(archive["potential_type"])
    well_amplitude = float(archive["well_amplitude"])

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
    x_flat = spatial_positions[..., 0].flatten()
    y_flat = spatial_positions[..., 1].flatten()

    box_length = 32.0 if potential_type == "chain" else None
    x_min, x_max = (0.0, box_length) if potential_type == "chain" else (-2.5, 2.5)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    # Trajectory Density or Multi-Trajectory Paths ---
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

    if n_trajectories > 1 or trajectory_ids is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        trajectory_stride = max(
            1, (spatial_positions.shape[1] + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS
        )
        for traj_idx in range(n_trajectories):
            ax1.plot(
                spatial_positions[traj_idx, ::trajectory_stride, 0],
                spatial_positions[traj_idx, ::trajectory_stride, 1],
                color=colors[traj_idx],
                alpha=0.7,
                linewidth=0.8,
                label=f"Traj {selected_ids[traj_idx]}"
                if n_trajectories <= 10
                else "",
            )
        ax1.set_title(
            f"2D Paths for Trajectories {', '.join(map(str, selected_ids))} over $V(x,y)$"
        )
        if n_trajectories <= 10:
            ax1.legend(loc="upper right", fontsize=7)
    else:
        hb = ax1.hexbin(spatial_positions[0, :, 0], spatial_positions[0, :, 1], gridsize=45, cmap="plasma", mincnt=1)
        fig.colorbar(hb, ax=ax1, label="Sample Density")
        ax1.set_title("2D Langevin Trajectory Density over $V(x,y)$")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # --- Energy Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(selected_energies.flatten(), bins=40, density=True, color="#6ba2c6", edgecolor="black", alpha=0.75)
    ax2.set_title("Boltzmann Energy Distribution")
    ax2.set_xlabel("Potential Energy $U(x, y)$")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ---  Marginal Distribution $p(x)$ & Barrier ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    ax3.hist(x_flat, bins=60, density=True, color="teal", alpha=0.6, label="Sample Density $p(x)$")
    x_line = np.linspace(x_min, x_max, 300)
    v_slice = compute_potential(
        x_line, np.zeros_like(x_line), potential_type, period=box_length or 32.0,
        well_amplitude=well_amplitude,
    )
    ax3_twin.plot(x_line, v_slice, color="black", lw=2, linestyle="--", label="$V(x, 0)$")
    ax3.set_title("Marginal Distribution $p(x)$ & Energy Barrier")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Probability Density", color="teal")
    ax3_twin.set_ylabel("Potential Energy $V(x, 0)$", color="black")

    # --- Panel 4: Time-Series (First Trajectory) ---
    ax4 = fig.add_subplot(gs[1, 1])
    x_traj_single = spatial_positions[0, :, 0]
    y_traj_single = spatial_positions[0, :, 1]
    steps = np.arange(len(x_traj_single))
    plot_count = max(1, int(0.1 * len(steps)))
    time_series_stride = max(1, (plot_count + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS)

    ax4.plot(
        steps[:plot_count:time_series_stride],
        x_traj_single[:plot_count:time_series_stride],
        label="x(t)",
        color="indigo",
        alpha=0.8,
        lw=0.8,
    )
    ax4.plot(
        steps[:plot_count:time_series_stride],
        y_traj_single[:plot_count:time_series_stride],
        label="y(t)",
        color="mediumseagreen",
        alpha=0.6,
        lw=0.8,
    )
    
    if potential_type == "chain":
        well_1, well_2 = (-1.0) % box_length, 1.0  # adjust if needed
    else:
        well_1, well_2 = -1.0, 1.0

    ax4.axhline(well_1, color="gray", linestyle=":", alpha=0.7, label="Well 1 Center")
    ax4.axhline(well_2, color="gray", linestyle=":", alpha=0.7, label="Well 2 Center")
    ax4.set_title(
        f"Trajectory {selected_ids[0]} Time-Series "
        f"(First {plot_count} saved frames)"
    )
    ax4.set_xlabel("Saved frames")
    ax4.set_ylabel("Position Coordinates")
    ax4.legend(loc="upper right", fontsize=8)

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
    args = parser.parse_args()
    plot_simulation(args.data, trajectory_ids=args.trajectories)





'''

    x_min, x_max = (0.0, box_length) if args.potential == "chain" else (-2.5, 2.5)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    contour = ax1.contour(X, Y, potential(X, Y), levels=12, cmap="Greys", alpha=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)

    # If multiple trajectories, plot them with different colors over time; if 1, show hexbin density
    if n_trajectories > 1:
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        for traj_idx in range(n_trajectories):
            ax1.plot(
                all_positions[traj_idx, :, 0],
                all_positions[traj_idx, :, 1],
                color=colors[traj_idx],
                alpha=0.7,
                linewidth=0.8,
                label=f"Traj {traj_idx + 1}" if n_trajectories <= 10 else "",
            )
        ax1.set_title(f"2D Paths for {n_trajectories} Trajectories over $V(x,y)$")
        if n_trajectories <= 10:
            ax1.legend(loc="upper right", fontsize=7)
    else:
        hb = ax1.hexbin(x_traj_arr, y_traj_arr, gridsize=45, cmap="plasma", mincnt=1)
        fig.colorbar(hb, ax=ax1, label="Sample Density")
        ax1.set_title("2D Langevin Trajectory Density over $V(x,y)$")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(gs[0, 1])
    # Flatten energies across all trajectories for the energy histogram
    ax2.hist(all_energies.flatten(), bins=40, density=True, color="#6ba2c6", edgecolor="black", alpha=0.75)
    ax2.set_title("Boltzmann Energy Distribution")
    ax2.set_xlabel("Potential Energy $U(x, y)$")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    ax3.hist(x_traj_arr, bins=60, density=True, color="teal", alpha=0.6, label="Sample Density $p(x)$")
    x_line = np.linspace(x_min, x_max, 300)
    ax3_twin.plot(x_line, potential(x_line, 0.0), color="black", lw=2, linestyle="--", label="$V(x, 0)$")
    ax3.set_title("Marginal Distribution $p(x)$ & Energy Barrier")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Probability Density", color="teal")
    ax3_twin.set_ylabel("Potential Energy $V(x, 0)$", color="black")

    ax4 = fig.add_subplot(gs[1, 1])
    steps = np.arange(len(x_traj_arr))
    plot_count = max(1, int(0.1 * len(steps)))

    ax4.plot(steps[:plot_count], x_traj_arr[:plot_count], label="x(t)", color="indigo", alpha=0.8, lw=0.8)
    ax4.plot(steps[:plot_count], y_traj_arr[:plot_count], label="y(t)", color="mediumseagreen", alpha=0.6, lw=0.8)
    if args.potential == "chain":
        well_1, well_2 = well_x_centers[0] % box_length, well_x_centers[1]
    else:
        well_1, well_2 = -1.0, 1.0
    ax4.axhline(well_1, color="gray", linestyle=":", alpha=0.7, label="Well 1 Center")
    ax4.axhline(well_2, color="gray", linestyle=":", alpha=0.7, label="Well 2 Center")
    ax4.set_title(f"Trajectory Time-Series (First {plot_count} saved frames)")
    ax4.set_xlabel(f"Saved frame (every {SAVE_EVERY} MD steps)")
    ax4.set_ylabel("Position Coordinates")
    ax4.legend(loc="upper right", fontsize=8)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / (
        f"viz_{args.dimension}D_{potential_name}_{scientific_count(n_steps)}steps_"
        f"{scientific_count(burn_in)}burn_{date}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

'''

    
    
    
    
    
    
    
'''
PREVIOUSLY IN THE END OF THE MD_SIM.PY SCRIPT - dont touch, delete or use - keep outcommented as archive.
    # Keep the existing four-panel visualization; only the x-domain follows the potential.
    x_min, x_max = (0.0, box_length) if args.potential == "chain" else (-2.5, 2.5)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    contour = ax1.contour(X, Y, potential(X, Y), levels=12, cmap="Greys", alpha=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)
    hb = ax1.hexbin(x_traj_arr, y_traj_arr, gridsize=45, cmap="plasma", mincnt=1)
    fig.colorbar(hb, ax=ax1, label="Sample Density")
    ax1.set_title("2D Langevin Trajectory Density over $V(x,y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_energies, bins=40, density=True, color="#6ba2c6", edgecolor="black", alpha=0.75)
    ax2.set_title("Boltzmann Energy Distribution")
    ax2.set_xlabel("Potential Energy $U(x, y)$")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    ax3.hist(x_traj_arr, bins=60, density=True, color="teal", alpha=0.6, label="Sample Density $p(x)$")
    x_line = np.linspace(x_min, x_max, 300)
    ax3_twin.plot(x_line, potential(x_line, 0.0), color="black", lw=2, linestyle="--", label="$V(x, 0)$")
    ax3.set_title("Marginal Distribution $p(x)$ & Energy Barrier")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Probability Density", color="teal")
    ax3_twin.set_ylabel("Potential Energy $V(x, 0)$", color="black")

    ax4 = fig.add_subplot(gs[1, 1])
    steps = np.arange(len(x_traj_arr))
    plot_count = max(1, int(0.1 * len(steps)))

    ax4.plot(steps[:plot_count], x_traj_arr[:plot_count], label="x(t)", color="indigo", alpha=0.8, lw=0.8)
    ax4.plot(steps[:plot_count], y_traj_arr[:plot_count], label="y(t)", color="mediumseagreen", alpha=0.6, lw=0.8)
    if args.potential == "chain":
        well_1, well_2 = well_x_centers[0] % box_length, well_x_centers[1]
    else:
        well_1, well_2 = -1.0, 1.0
    ax4.axhline(well_1, color="gray", linestyle=":", alpha=0.7, label="Well 1 Center")
    ax4.axhline(well_2, color="gray", linestyle=":", alpha=0.7, label="Well 2 Center")
    ax4.set_title(f"Trajectory Time-Series (First {plot_count} saved frames)")
    ax4.set_xlabel(f"Saved frame (every {SAVE_EVERY} MD steps)")
    ax4.set_ylabel("Position Coordinates")
    ax4.legend(loc="upper right", fontsize=8)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / (
        f"viz_{args.dimension}D_{potential_name}_{scientific_count(n_steps)}steps_"
        f"{scientific_count(burn_in)}burn_{date}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(
        f"Saved {positions.shape} positions to {data_path} "
        f"(every {SAVE_EVERY} MD steps; interval {SAVE_EVERY * dt:g})"
    )
    if target_transition_rate is not None:
        print(f"Target harmonic-TST rate: {target_transition_rate:.6g} per time unit")
        print(f"Expected transitions in first half ({half_production_time:.6g} time units): {args.target_transitions_half:g}")
        print(f"TST barrier above the left-well minimum: {target_barrier:.6g}")
        print(f"Adjusted Gaussian-well amplitude: {well_amplitude:.6g}")
    print(f"Saved plot to {plot_path}")

'''
