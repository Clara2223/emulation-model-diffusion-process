import argparse
from datetime import datetime
from pathlib import Path
import numba as nb
from numba import jit

import matplotlib.pyplot as plt
import numpy as np

from TST_rate import harmonic_tst, well_amplitude_for_rate


SAVE_EVERY = 10


def scientific_count(value):
    digits = str(int(value))
    significant = digits[1:].rstrip("0")
    return f"{digits[0]}{'.' + significant if significant else ''}e{len(digits) - 1}"


def main():
    parser = argparse.ArgumentParser(description="Generate Langevin MD training data.")
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2,
                        help="Save x,y (2D) or x,y plus normalized time (3D).")
    parser.add_argument(
        "--potential",
        choices=("double-well", "double-well-adjusted", "double-well-Hbarrier", "chain"),
        default="double-well",
    )
    parser.add_argument("--n-steps", type=int, default=100_000_000)
    parser.add_argument("--burn-in", type=int, default=50_000)
    parser.add_argument(
        "--well-amplitude", type=float,
        help="Depth of the Gaussian wells for a double-well potential (default: 3).",
    )
    parser.add_argument("--target-transitions-half", type=float, default=1.0,
                        help="Expected TST transitions in the first half of saved MD.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "train_data")
    parser.add_argument("--plot-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if args.n_steps <= args.burn_in:
        parser.error("--n-steps must be greater than --burn-in")
    if args.well_amplitude is not None and args.well_amplitude <= 0:
        parser.error("--well-amplitude must be positive")
    if args.target_transitions_half <= 0:
        parser.error("--target-transitions-half must be positive")
    is_chain = args.potential == "chain"
    # Reduced-unit Langevin parameters; ål the double well retains its confining spring
    np.random.seed(3)
    dt, mass, gamma, kBT = 0.005, 1.0, 1.0, 0.4
    n_steps, burn_in = args.n_steps, args.burn_in
    A1, A2, sigma_x, sigma_y = 3.0, 3.0, 0.5, 0.5
    k_confine = k_y_confinement = 0.2
    well_amplitude = 3.0 if args.well_amplitude is None else args.well_amplitude
    target_transition_rate = None
    half_production_time = None
    target_barrier = None
    if args.potential in {"double-well-adjusted", "double-well-Hbarrier"}:
        if args.well_amplitude is not None:
            parser.error("The adjusted double-well solves --well-amplitude from the target rate")
        half_production_time = 0.5 * (n_steps - burn_in) * dt
        target_transition_rate = args.target_transitions_half / half_production_time
        tst_parameters = {
            "kBT": kBT,
            "mass": mass,
            "well_width": sigma_x,
            "confinement": k_confine,
        }
        well_amplitude = well_amplitude_for_rate(
            target_transition_rate, **tst_parameters,
        )
        _, target_barrier = harmonic_tst(well_amplitude, **tst_parameters)
    elif is_chain and args.well_amplitude is not None:
        parser.error("--well-amplitude applies only to double-well potentials")
    if not is_chain:
        A1 = well_amplitude
    well_separation, cell_spacing, n_cells = 2.0, 4.0, 8
    box_length = n_cells * cell_spacing
    cell_centers = cell_spacing * np.arange(n_cells)
    well_x_centers = (cell_centers[:, None] + [-well_separation / 2, well_separation / 2]).ravel()
    well_amplitudes = np.tile([A1, A2], n_cells)
    double_well_centers = np.array([[-1.0, 0.0], [1.0, 0.0]])

    @jit(nopython=True)
    def potential(x, y):
        x, y = np.asarray(x), np.asarray(y)
        
        if is_chain:
            dx = (x[..., None] - well_x_centers + box_length / 2) % box_length - box_length / 2
            dy = y[..., None]
            wells = well_amplitudes * np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
            return -np.sum(wells, axis=-1) + 0.5 * k_y_confinement * y**2
        
        dx, dy = x[..., None] - double_well_centers[:, 0], y[..., None] - double_well_centers[:, 1]
        wells = A1 * np.exp(-0.5 * (dx**2 + dy**2) / sigma_x**2)
        return -np.sum(wells, axis=-1) + 0.5 * k_confine * (x**2 + y**2)

    @jit(nopython=True)
    def forces(x, y):
        
        if is_chain:
            dx = (x - well_x_centers + box_length / 2) % box_length - box_length / 2
            dy = y
            wells = well_amplitudes * np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
            return -np.sum(wells * dx / sigma_x**2), -np.sum(wells * dy / sigma_y**2) - k_y_confinement * y
        
        dx, dy = x - double_well_centers[:, 0], y - double_well_centers[:, 1]
        wells = A1 * np.exp(-0.5 * (dx**2 + dy**2) / sigma_x**2)
        return (-np.sum(wells * dx / sigma_x**2) - k_confine * x,
                -np.sum(wells * dy / sigma_y**2) - k_confine * y)

    # Underdamped Langevin dynamics; only the chain wraps periodically in x.
    position = np.array([(-well_separation / 2 if args.potential == "chain" else -1.0), 0.0])
    velocity = np.zeros(2)
    noise_scale = np.sqrt(2 * gamma * kBT * dt) / mass
    n_saved = (n_steps - burn_in + SAVE_EVERY - 1) // SAVE_EVERY
    positions = np.empty((n_saved, 2), dtype=np.float32)
    velocities = np.empty((n_saved, 2), dtype=np.float32)
    energy_traj = np.empty(n_saved, dtype=np.float32)
    
    for step in range(n_steps):
        force = np.asarray(forces(position[0], position[1]))
        velocity += (force - gamma * velocity) * (dt / mass) + noise_scale * np.random.normal(size=2)
        position += velocity * dt

        if args.potential == "chain":
            position[0] %= box_length

        if step >= burn_in and (step - burn_in) % SAVE_EVERY == 0:
            output_index = (step - burn_in) // SAVE_EVERY
            positions[output_index] = position
            velocities[output_index] = velocity
            energy_traj[output_index] = potential(position[0], position[1])

    md_time = np.arange(burn_in, n_steps, SAVE_EVERY) * dt
    x_traj_arr, y_traj_arr = positions[:, 0], positions[:, 1]

    if args.dimension == 3:
        positions = np.column_stack((positions, np.linspace(0, 1, len(positions), dtype=np.float32)))

    date = datetime.now().strftime("%Y%m%d_%H%M")
    potential_name = (
        "double_well_adjusted"
        if args.potential in {"double-well-adjusted", "double-well-Hbarrier"}
        else args.potential.replace("-", "_")
    )

    data_path = args.output_dir / (
        f"{args.dimension}D_{potential_name}_{scientific_count(n_steps)}steps_"
        f"{scientific_count(burn_in)}burn_{date}.npz"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, positions=positions, md_time=md_time,
                        velocities=velocities, energy=energy_traj,
                        save_stride=SAVE_EVERY,
                        potential_type="double-well" if not is_chain else "chain",
                        well_amplitude=A1)

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
    ax2.hist(energy_traj, bins=40, density=True, color="#6ba2c6", edgecolor="black", alpha=0.75)
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


if __name__ == "__main__":
    main()