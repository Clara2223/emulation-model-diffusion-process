import argparse
from datetime import datetime
from pathlib import Path
import numba as nb
from numba import jit

import matplotlib.pyplot as plt
import numpy as np

from .TST_rate import harmonic_tst, well_amplitude_for_rate
from .plot import plot_simulation
from .potentials import compute_potential

SAVE_EVERY = 10


def scientific_count(value):
    digits = str(int(value))
    significant = digits[1:].rstrip("0")
    return f"{digits[0]}{'.' + significant if significant else ''}e{len(digits) - 1}"


def main():
    parser = argparse.ArgumentParser(description="Generate Langevin MD training data")
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2,
                        help="Save x,y (2D) or x,y plus normalized time (3D)")
    parser.add_argument(
        "--potential",
        choices=("double-well", "double-well-adjusted", "double-well-Hbarrier", "chain"),
        default="double-well",
    )
    parser.add_argument("--n-trajectories", type=int, default=1, help="Number of independent MD trajectories to simulate")
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
    if args.n_trajectories < 1:
        parser.error("--n-trajectories must be positive")
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
        A2 = well_amplitude
    
    
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
    
    n_trajectories = args.n_trajectories
    n_saved = (n_steps - burn_in + SAVE_EVERY - 1) // SAVE_EVERY
    
    all_positions = np.empty((n_trajectories, n_saved, 2), dtype=np.float32)
    all_velocities = np.empty((n_trajectories,n_saved, 2), dtype=np.float32)
    all_energies = np.empty((n_trajectories, n_saved), dtype=np.float32)
    initial_positions = np.empty((n_trajectories, 2), dtype=np.float32)
    
    for traj in range(n_trajectories):
        # Init x position
        start_x = -well_separation / 2 if args.potential == "chain" else -1.0
        position = np.array([start_x, 0.0])
        velocity = np.zeros(2)
        
        # random perturbation for different init cond. per trajectory
        #position += np.random.normal(0, 1, size=2)
        initial_positions[traj] = position.copy()  # Save true init pos. of this trajectory

        traj_positions = np.empty((n_saved, 2), dtype=np.float32)
        traj_velocities = np.empty((n_saved, 2), dtype=np.float32)
        traj_energies = np.empty(n_saved, dtype=np.float32)

        for step in range(n_steps):
            force = np.asarray(forces(position[0], position[1]))
            velocity += (force - gamma * velocity) * (dt / mass) + noise_scale * np.random.normal(size=2)
            position += velocity * dt

            if args.potential == "chain":
                position[0] %= box_length

            if step >= burn_in and (step - burn_in) % SAVE_EVERY == 0:
                output_index = (step - burn_in) // SAVE_EVERY
                traj_positions[output_index] = position
                traj_velocities[output_index] = velocity
                traj_energies[output_index] = potential(position[0], position[1])

        all_positions[traj] = traj_positions
        all_velocities[traj] = traj_velocities
        all_energies[traj] = traj_energies
    
    md_time = np.arange(burn_in, n_steps, SAVE_EVERY) * dt
    

    positions = all_positions
    if args.dimension == 3:
        normalized_time = np.broadcast_to(
            np.linspace(0, 1, n_saved, dtype=np.float32)[None, :],
            (n_trajectories, n_saved),
        )
        positions = np.concatenate((positions, normalized_time[..., None]), axis=2)

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
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / f"viz_{data_path.stem}.png"
    
    np.savez_compressed(
        data_path,
        positions=positions,                 # shape: (n_trajectories, n_saved, 2 or 3)
        initial_positions=initial_positions, # Shape: (n_trajectories, 2)
        md_time=md_time,
        velocities=all_velocities,
        energy=all_energies,
        save_stride=SAVE_EVERY,
        potential_type="double-well" if not is_chain else "chain",
        well_amplitude=A1,
    )
    


    plot_simulation(data_path, plot_path)

if __name__ == "__main__":
    main()