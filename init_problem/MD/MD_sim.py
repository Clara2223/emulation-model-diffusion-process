import argparse
from datetime import datetime
from pathlib import Path
import numba as nb
from numba import jit

import matplotlib.pyplot as plt
import numpy as np


def scientific_count(value):
    digits = str(int(value))
    significant = digits[1:].rstrip("0")
    return f"{digits[0]}{'.' + significant if significant else ''}e{len(digits) - 1}"


def estimate_hbarrier_amplitude(
    n_steps,
    burn_in,
    dt,
    kBT,
    mass,
    target_transitions,
    barrier_width,
    well_amplitude=3.0,
    well_width=0.5,
    confinement=0.2,
):
    """Choose a Gaussian ridge height for the target harmonic-TST event count."""
    if not 0 < barrier_width or target_transitions <= 0:
        raise ValueError("Barrier width and target transition count must be positive")

    half_production_time = 0.5 * (n_steps - burn_in) * dt
    target_rate = target_transitions / half_production_time

    def values(x, ridge_amplitude):
        offsets = (x + 1.0, x - 1.0)
        gaussians = tuple(
            np.exp(-offset**2 / (2 * well_width**2)) for offset in offsets
        )
        ridge = np.exp(-x**2 / (2 * barrier_width**2))
        energy = (
            -well_amplitude * sum(gaussians)
            + 0.5 * confinement * x**2
            + ridge_amplitude * ridge
        )
        gradient = (
            well_amplitude / well_width**2
            * sum(offset * gaussian for offset, gaussian in zip(offsets, gaussians))
            + confinement * x
            - ridge_amplitude * x / barrier_width**2 * ridge
        )
        curvature = (
            well_amplitude / well_width**2
            * sum(
                gaussian * (1 - offset**2 / well_width**2)
                for offset, gaussian in zip(offsets, gaussians)
            )
            + confinement
            + ridge_amplitude * ridge
            * (x**2 / barrier_width**4 - 1 / barrier_width**2)
        )
        transverse_curvature = (
            well_amplitude / well_width**2 * sum(gaussians) + confinement
        )
        return energy, gradient, curvature, transverse_curvature

    def estimate_rate(ridge_amplitude):
        well_x = -1.0
        for _ in range(50):
            _, gradient, curvature, _ = values(well_x, ridge_amplitude)
            if curvature <= 0:
                raise ValueError("The proposed ridge removes the left-well minimum")
            
            
            
            step = gradient / curvature
            well_x -= step
            
            if abs(step) < 1e-12:
                break
        else:
            raise ValueError("Could not locate the left-well minimum")

        well_energy, _, well_x_curvature, well_y_curvature = values(
            well_x, ridge_amplitude,
        )
        saddle_energy, _, saddle_x_curvature, saddle_y_curvature = values(
            0.0, ridge_amplitude,
        )
        if saddle_x_curvature >= 0 or saddle_y_curvature <= 0:
            raise ValueError("The added ridge does not produce a first-order saddle")

        barrier = saddle_energy - well_energy
        
        prefactor = (
            np.sqrt(well_x_curvature / mass)
            * np.sqrt(well_y_curvature / mass)
            / np.sqrt(saddle_y_curvature / mass)
            / (2 * np.pi)
        )
        return prefactor * np.exp(-barrier / kBT), barrier

    low, high = 0.0, 1.0
    low_rate, _ = estimate_rate(low)
    if low_rate <= target_rate:
        raise ValueError("The unmodified double well is already slower than the target rate")
    while estimate_rate(high)[0] > target_rate:
        high *= 2
        if high > 1_000:
            raise ValueError("Could not find a ridge height for the target transition rate")

    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if estimate_rate(midpoint)[0] > target_rate:
            low = midpoint
        else:
            high = midpoint

    amplitude = 0.5 * (low + high)
    rate, barrier = estimate_rate(amplitude)
    return amplitude, barrier, rate, half_production_time


def main():
    parser = argparse.ArgumentParser(description="Generate Langevin MD training data.")
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2,
                        help="Save x,y (2D) or x,y plus normalized time (3D).")
    parser.add_argument("--potential", choices=("double-well","double-well-Hbarrier" ,"chain"), default="double-well")
    parser.add_argument("--n-steps", type=int, default=100_000_000)
    parser.add_argument("--burn-in", type=int, default=50_000)
    parser.add_argument("--barrier-width", type=float, default=0.25,
                        help="Gaussian ridge width in x for double-well-Hbarrier.")
    parser.add_argument("--target-transitions-half", type=float, default=1.0,
                        help="Expected TST transitions in the first half of saved MD.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "train_data")
    parser.add_argument("--plot-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if args.n_steps <= args.burn_in:
        parser.error("--n-steps must be greater than --burn-in")
    is_chain = args.potential == "chain"
    barrier_width = args.barrier_width

    # Reduced-unit Langevin parameters; ål the double well retains its confining spring
    np.random.seed(3)
    dt, mass, gamma, kBT = 0.005, 1.0, 1.0, 0.4
    n_steps, burn_in = args.n_steps, args.burn_in
    A1, A2, sigma_x, sigma_y = 3.0, 3.0, 0.5, 0.5
    k_confine = k_y_confinement = 0.2
    barrier_amplitude = 0.0
    target_transition_rate = None
    half_production_time = None
    target_barrier = None
    if args.potential == "double-well-Hbarrier":
        (
            barrier_amplitude,
            target_barrier,
            target_transition_rate,
            half_production_time,
        ) = estimate_hbarrier_amplitude(
            n_steps,
            burn_in,
            dt,
            kBT,
            mass,
            args.target_transitions_half,
            barrier_width,
            well_amplitude=A1,
            well_width=sigma_x,
            confinement=k_confine,
        )
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
        ridge = barrier_amplitude * np.exp(-0.5 * (x / barrier_width) ** 2)
        return -np.sum(wells, axis=-1) + 0.5 * k_confine * (x**2 + y**2) + ridge

    @jit(nopython=True)
    def forces(x, y):
        
        if is_chain:
            dx = (x - well_x_centers + box_length / 2) % box_length - box_length / 2
            dy = y
            wells = well_amplitudes * np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
            return -np.sum(wells * dx / sigma_x**2), -np.sum(wells * dy / sigma_y**2) - k_y_confinement * y
        
        dx, dy = x - double_well_centers[:, 0], y - double_well_centers[:, 1]
        wells = A1 * np.exp(-0.5 * (dx**2 + dy**2) / sigma_x**2)
        ridge_force = (
            barrier_amplitude * x / barrier_width**2
            * np.exp(-0.5 * (x / barrier_width) ** 2)
        )
        return (-np.sum(wells * dx / sigma_x**2) - k_confine * x + ridge_force,
                -np.sum(wells * dy / sigma_y**2) - k_confine * y)

    # Underdamped Langevin dynamics; only the chain wraps periodically in x.
    position = np.array([(-well_separation / 2 if args.potential == "chain" else -1.0), 0.0])
    velocity = np.zeros(2)
    noise_scale = np.sqrt(2 * gamma * kBT * dt) / mass
    x_traj, y_traj, vx_traj, vy_traj, energy_traj = [], [], [], [], []
    
    '''    
    for step in range(n_steps):
        fx, fy = forces(x, y)
        vx += (fx / mass - gamma * vx / mass) * dt + noise_scale * np.random.normal()
        vy += (fy / mass - gamma * vy / mass) * dt + noise_scale * np.random.normal()
        x, y = x + vx * dt, y + vy * dt
        
        if args.potential == "chain":
            x %= box_length
        
        if step >= burn_in:
            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(vx)
            vy_traj.append(vy)
            energy_traj.append(potential(x, y))
    '''
    
    n_saved = n_steps - burn_in
    positions = np.empty((n_saved, 2), dtype=np.float32)
    velocities = np.empty((n_saved, 2), dtype=np.float32)
    energy_traj = np.empty(n_saved, dtype=np.float32)
    
    for step in range(n_steps):
        force = np.asarray(forces(position[0], position[1]))
        velocity += (force - gamma * velocity) * (dt / mass) + noise_scale * np.random.normal(size=2)
        position += velocity * dt

        if args.potential == "chain":
            position[0] %= box_length

        if step >= burn_in:
            output_index = step - burn_in
            positions[output_index] = position
            velocities[output_index] = velocity
            energy_traj[output_index] = potential(position[0], position[1])

    md_time = np.arange(burn_in, n_steps) * dt
    x_traj_arr, y_traj_arr = positions[:, 0], positions[:, 1]

    if args.dimension == 3:
        positions = np.column_stack((positions, np.linspace(0, 1, len(positions), dtype=np.float32)))

    date = datetime.now().strftime("%Y%m%d_%H%M")
    potential_name = args.potential.replace("-", "_")

    data_path = args.output_dir / (
        f"{args.dimension}D_{potential_name}_{scientific_count(n_steps)}steps_"
        f"{scientific_count(burn_in)}burn_{date}.npz"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, positions=positions, md_time=md_time,
                        velocities=velocities, energy=energy_traj,
                        potential_type=args.potential,
                        barrier_amplitude=barrier_amplitude,
                        barrier_width=barrier_width)

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
    #plot_steps = min(burn_in, len(steps))
    #ax4.plot(steps[:burn_in], x_traj_arr[:burn_in], label="x(t)", color="indigo", alpha=0.8, lw=0.8)
    #ax4.plot(steps[:burn_in], y_traj_arr[:burn_in], label="y(t)", color="mediumseagreen", alpha=0.6, lw=0.8)
    ax4.plot(steps[:plot_count], x_traj_arr[:plot_count], label="x(t)", color="indigo", alpha=0.8, lw=0.8)
    ax4.plot(steps[:plot_count], y_traj_arr[:plot_count], label="y(t)", color="mediumseagreen", alpha=0.6, lw=0.8)
    if args.potential == "chain":
        well_1, well_2 = well_x_centers[0] % box_length, well_x_centers[1]
    else:
        well_1, well_2 = -1.0, 1.0
    ax4.axhline(well_1, color="gray", linestyle=":", alpha=0.7, label="Well 1 Center")
    ax4.axhline(well_2, color="gray", linestyle=":", alpha=0.7, label="Well 2 Center")
    ax4.set_title(f"Trajectory Time-Series (First {plot_count} saved steps)")
    ax4.set_xlabel("Saved Step")
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
    print(f"Saved {positions.shape} positions to {data_path}")
    if target_transition_rate is not None:
        print(f"Target harmonic-TST rate: {target_transition_rate:.6g} per time unit")
        print(f"Expected transitions in first half ({half_production_time:.6g} time units): {args.target_transitions_half:g}")
        print(f"TST barrier above the left-well minimum: {target_barrier:.6g}")
        print(f"Added Gaussian ridge amplitude: {barrier_amplitude:.6g}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()