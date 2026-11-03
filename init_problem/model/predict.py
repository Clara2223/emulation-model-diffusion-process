import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from MD.potentials import CHAIN_PERIOD, compute_potential as potential
#from ..MD.potentials import CHAIN_PERIOD, compute_potential as potential
from .data import STATE_DIM
from .model_2D import DEVICE, SIGMA_MIN


LOG_FIELDS = ["timestamp",
    "png",
    "model_version",
    "einstein_diffusion",
    "kramers_barrier_height",
    "kramers_transition_rate",
    "sampling_config",
    "histogram_marginal_hellinger",
    "histogram_joint_hellinger",
    "reference_crossing_rate",
    "generated_crossing_rate",
    "device",
]


#diffusion model predictions
#generating *equilibrium samples* with annealed (discrete steps) Langevin dynamics
def sample_score(model, mean, std, n_samples=1_000, n_levels=12, steps_per_level=50):
    model.eval()
    X = torch.randn(n_samples, STATE_DIM, device=DEVICE)
    #this sigma sampling is the same as in training but reversed: from high noise to low noise
    sigmas = torch.exp(torch.linspace(np.log(1.0), np.log(0.02), n_levels, device=DEVICE))
    
    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = torch.full((n_samples,), sigma, device=DEVICE)
            step_size = 0.02 * sigma**2
            
            for _ in range(steps_per_level):
                # Euler-Maruyama Langevin step: score drift plus Gaussian noise
                X = X + step_size * model(X, sigma_batch)
                X = X + torch.sqrt(2 * step_size) * torch.randn_like(X)
    
    return X.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()

#flow model predictions
def sample_flow(model, mean, std, n_samples=1_000, ode_steps=50):
    """Generate samples by integrating the learned flow from noise to data."""
    model.eval()
    X = torch.randn(n_samples, STATE_DIM, device=DEVICE)
    step_size = 1.0 / ode_steps

    with torch.no_grad():
        for step in range(ode_steps):
            tau = torch.full((n_samples,), step / ode_steps, device=DEVICE)
            X = X + step_size * model(X, tau)

    return X.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()



def calculate_einstein_diffusion(positions, times, dt=0.005):
    """ 
    Einstein connected microscopic brownian motion to (macroscopic) diffusion.
    - Ficks second law: prob. p over time: dp/dt= D nabla**2 p
    - mean square displacement: avg. square distance x grow grow linearly in time : <x**2>=2Dt (1D)
    - Stokes-Einstein equation: for spherial particle in fluid w dynamic viscosity eta at absolute temp. T: D=k_BT/6pi*eta*a
    - Einstein relation (diffusion): connects particle mobility mu (drift velocity per unit force) to D: D=mu*k_BT (mu_q*k_BT / q for EM/charged particles)
    
    BELOW: D is estimated from the mean square displacement (MSD) of trajectory
    from the 'late-time' slope of the MSD
    """
    displacement = positions - positions[0]
    time = np.asarray(times)
    msd = np.sum(displacement**2, axis=1)
    slope, _ = np.polyfit(time[len(time) // 2 :], msd[len(time) // 2 :], 1)     #aka msd/t
    return slope / 4.0                                      # 4 because 2D


def calculate_kramers_rate(
    kBT=0.4,
    mass=1.0,
    gamma=1.0,
    potential_type="double-well",
    barrier_amplitude=0.0,
    barrier_width=0.25,
    period=CHAIN_PERIOD,
    well_amplitude=3.0,
):
    """Estimate the underdamped barrier-crossing rate for this potential."""
    dx = 1e-4
    energy = lambda x: potential(
        x, 0.0, potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    well_x = -1.0
    for _ in range(20):
        first_derivative = (energy(well_x + dx) - energy(well_x - dx)) / (2 * dx)
        second_derivative = (
            energy(well_x + dx) - 2 * energy(well_x) + energy(well_x - dx)
        ) / dx**2
        if second_derivative <= 0:
            raise ValueError("Could not locate a stable well for the Kramers estimate")
        step = first_derivative / second_derivative
        well_x -= step
        if abs(step) < 1e-10:
            break
    well = energy(well_x)
    barrier = energy(0.0)
    barrier_height = barrier - well
    well_curvature = (
        energy(well_x + dx)
        - 2 * well
        + energy(well_x - dx)
    ) / dx**2
    
    barrier_curvature = (
        energy(dx) - 2 * barrier + energy(-dx)
    ) / dx**2
    
    omega_well = np.sqrt(well_curvature / mass)
    omega_barrier = np.sqrt(abs(barrier_curvature) / mass)
    
    transmission = np.sqrt(1 + (gamma / (2 * omega_barrier)) ** 2) - gamma / (
        2 * omega_barrier
    )
    
    rate = transmission * omega_well / (2 * np.pi) * np.exp(-barrier_height / kBT)
    
    return rate, barrier_height


def histogram_distances(reference, generated, bins=32):
    """Return mean marginal and full-joint Hellinger distances on common sample ranges."""
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if (
        reference.ndim != 2
        or generated.ndim != 2
        or reference.shape[1] != generated.shape[1]
        or not len(reference)
        or not len(generated)
    ):
        raise ValueError("Histogram comparison requires non-empty samples with matching dimensions")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("Histogram comparison samples must be finite")

    lower = np.minimum(reference.min(axis=0), generated.min(axis=0))
    upper = np.maximum(reference.max(axis=0), generated.max(axis=0))
    span = upper - lower
    lower = np.where(span == 0, lower - 0.5, lower)
    upper = np.where(span == 0, upper + 0.5, upper)
    edges = [np.linspace(lo, hi, bins + 1) for lo, hi in zip(lower, upper)]

    def hellinger(first, second):
        first = first / first.sum()
        second = second / second.sum()
        return float(np.sqrt(0.5 * np.square(np.sqrt(first) - np.sqrt(second)).sum()))

    marginal = np.mean([
        hellinger(
            np.histogram(reference[:, dim], bins=edges[dim])[0],
            np.histogram(generated[:, dim], bins=edges[dim])[0],
        )
        for dim in range(reference.shape[1])
    ])
    joint = hellinger(
        np.histogramdd(reference, bins=edges)[0],
        np.histogramdd(generated, bins=edges)[0],
    )
    return float(marginal), joint


#log file
def log_prediction(
    output_path,
    model_version,
    einstein_diffusion,
    barrier_height,
    kramers_rate,
    device=DEVICE,
    sampling_config="",
    histogram_marginal_hellinger=None,
    histogram_joint_hellinger=None,
    reference_crossing_rate=None,
    generated_crossing_rate=None,
):
    """Append run metadata and register older PNGs without known metadata.

    The two histogram distances are in [0, 1], where 0 is identical and 1 is
    maximally different. Marginal distance averages the per-coordinate
    histograms; joint distance compares the full multivariate histogram.
    """
    output_path = Path(output_path)
    log_path = output_path.parent / "prediction_log.csv"
    existing_rows = []
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as log_file:
            content = (line for line in log_file if not line.startswith("#"))
            reader = csv.DictReader(content)
            if reader.fieldnames and "png" in reader.fieldnames:
                existing_rows = [
                    {field: row.get(field, "") for field in LOG_FIELDS}
                    for row in reader
                ]

    current_png = output_path.name
    known_pngs = {row["png"] for row in existing_rows}
    for png_path in sorted(output_path.parent.glob("*.png")):
        relative_png = png_path.name
        if relative_png.endswith("_loss.png"):
            continue
        if relative_png != current_png and relative_png not in known_pngs:
            existing_rows.append(
                {
                    "timestamp": "",
                    "png": relative_png,
                    "model_version": "unknown_existing",
                    "einstein_diffusion": "",
                    "kramers_barrier_height": "",
                    "kramers_transition_rate": "",
                    "sampling_config": "",
                    "histogram_marginal_hellinger": "",
                    "histogram_joint_hellinger": "",
                    "reference_crossing_rate": "",
                    "generated_crossing_rate": "",
                    "device": "",
                }
            )

    if current_png not in {row["png"] for row in existing_rows}:
        existing_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "png": current_png,
                "model_version": model_version,
                "einstein_diffusion": (
                    "" if einstein_diffusion is None else f"{einstein_diffusion:.8f}"
                ),
                "kramers_barrier_height": (
                    "" if barrier_height is None else f"{barrier_height:.8f}"
                ),
                "kramers_transition_rate": (
                    "" if kramers_rate is None else f"{kramers_rate:.8f}"
                ),
                "sampling_config": sampling_config,
                "histogram_marginal_hellinger": (
                    "" if histogram_marginal_hellinger is None
                    else f"{histogram_marginal_hellinger:.8f}"
                ),
                "histogram_joint_hellinger": (
                    "" if histogram_joint_hellinger is None
                    else f"{histogram_joint_hellinger:.8f}"
                ),
                "reference_crossing_rate": (
                    "" if reference_crossing_rate is None
                    else f"{reference_crossing_rate:.8f}"
                ),
                "generated_crossing_rate": (
                    "" if generated_crossing_rate is None
                    else f"{generated_crossing_rate:.8f}"
                ),
                "device": device,
            }
        )

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        log_file.write(
            "# histogram_marginal_hellinger: mean per-coordinate Hellinger distance "
            "between reference and generated sample histograms; 0=identical, 1=maximal.\n"
            "# histogram_joint_hellinger: Hellinger distance between full "
            "multivariate histograms; 0=identical, 1=maximal.\n"
            "# crossing_rate: basin-label changes per unit of physical MD time; "
            "defined for trajectory-model runs only.\n"
        )
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
    return log_path


#plotting
def plot_training_loss(
    losses,
    output_path,
    validation_losses=None,
    show=False,
    loss_curves="both",
):
    if loss_curves not in {"both", "training", "validation"}:
        raise ValueError(f"Unsupported loss-curve selection: {loss_curves}")
    if loss_curves in {"both", "training"} and not losses:
        raise ValueError("Training loss history is required for the selected curves")
    if loss_curves in {"both", "validation"} and not validation_losses:
        raise ValueError("Validation loss history is required for the selected curves")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    if loss_curves in {"both", "training"}:
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", linewidth=1.5, label="Training loss")
    if loss_curves in {"both", "validation"}:
        validation_epochs = np.arange(1, len(validation_losses) + 1)
        ax.plot(
            validation_epochs, validation_losses, marker="o", linewidth=1.5,
            label="Validation loss",
        )
    ax.set_title("Training and Validation Loss by Epoch" if loss_curves == "both" else f"{loss_curves.title()} Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_comparison(
    reference,
    generated,
    reference_time,
    generated_time,
    objective,
    output,
    md_stride=1,
    generated_stride=1,
    stop_time=None,
    trajectory_positions=None,
    trajectory_times=None,
    model=None,
    model_type=None,
    field_config=None,
    potential_type="double-well",
    barrier_amplitude=0.0,
    barrier_width=0.25,
    period=CHAIN_PERIOD,
    well_amplitude=3.0,
    training_positions=None,
    field_time=None,
    field_time_secondary=None,
    early_samples=None,
    early_time=None,
    late_samples=None,
    late_time=None,
    field_initial_position=None,
):
    reference = np.asarray(reference)
    reference_time = np.asarray(reference_time)
    generated = np.asarray(generated)
    generated_time = np.asarray(generated_time)
    if reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("reference samples must contain x,y coordinates")
    if generated.ndim != 2 or generated.shape[1] < 2:
        raise ValueError("generated samples must contain x,y coordinates")
    if len(reference) != len(reference_time) or len(generated) != len(generated_time):
        raise ValueError("Sample arrays and their physical-time arrays must match")
    if not len(reference) or not len(generated) or not len(reference_time):
        raise ValueError("Reference/generated samples and times must be non-empty")
    if md_stride < 1 or generated_stride < 1:
        raise ValueError("Plot strides must be at least 1")
    if trajectory_positions is None or trajectory_times is None:
        raise ValueError("trajectory_positions and trajectory_times are required")
    trajectory_positions = np.asarray(trajectory_positions)
    trajectory_times = np.asarray(trajectory_times)
    if (
        trajectory_positions.ndim != 3
        or trajectory_positions.shape[2] < 2
        or trajectory_times.ndim != 1
        or trajectory_positions.shape[1] != len(trajectory_times)
    ):
        raise ValueError("Trajectory positions and times must have matching shapes")
    trajectory_positions = trajectory_positions[:, :, :2].copy()
    if model_type not in {None, "diffusion", "flow"}:
        raise ValueError(f"Unsupported model type for plotting: {model_type}")
    if field_initial_position is None:
        field_initial_position = trajectory_positions[0, 0, :2]

    reference_time_bounds = (reference_time[0], reference_time[-1])
    reference = reference[::md_stride]
    reference_time = reference_time[::md_stride]
    generated = generated[::generated_stride]
    generated_time = generated_time[::generated_stride]
    if stop_time is not None:
        if (
            not np.isfinite(stop_time)
            or stop_time < reference_time_bounds[0]
            or stop_time > reference_time_bounds[1]
        ):
            raise ValueError(
                f"stop_time must be between {reference_time_bounds[0]:g} and "
                f"{reference_time_bounds[1]:g}"
            )
        reference_mask = reference_time <= stop_time
        generated_mask = generated_time <= stop_time
        if not reference_mask.any() or not generated_mask.any():
            raise ValueError("stop_time excludes all reference or generated samples")
        reference = reference[reference_mask]
        reference_time = reference_time[reference_mask]
        generated = generated[generated_mask]
        generated_time = generated_time[generated_mask]

    reference_stride = max(1, (len(reference) + 199_999) // 200_000)
    reference = reference[::reference_stride]
    reference_time = reference_time[::reference_stride]
    generated_plot_stride = max(1, (len(generated) + 199_999) // 200_000)
    generated = generated[::generated_plot_stride]
    generated_time = generated_time[::generated_plot_stride]
    if training_positions is None:
        training_positions = trajectory_positions
    raw_training_positions = np.asarray(training_positions)
    if raw_training_positions.ndim not in {2, 3} or raw_training_positions.shape[-1] < 2:
        raise ValueError("training_positions must contain x,y samples")
    if not len(reference) or not len(generated) or not raw_training_positions.size:
        raise ValueError("Plotting requires non-empty reference, generated, and training samples")
    if generated.ndim != 2 or generated.shape[1] < 2:
        raise ValueError("generated samples must contain x,y coordinates")

    x_min, x_max = (0.0, period) if potential_type == "chain" else (-3.5, 3.5)
    total_training_count = int(np.prod(raw_training_positions.shape[:-1]))

    def get_training_rows(indices):
        if raw_training_positions.ndim == 3:
            frame_count = raw_training_positions.shape[1]
            rows = raw_training_positions[
                indices // frame_count, indices % frame_count, :2,
            ]
        else:
            rows = raw_training_positions[indices, :2]
        rows = np.asarray(rows, dtype=np.float64)
        if potential_type == "chain":
            rows[:, 0] %= period
        return rows

    sample_indices = np.linspace(
        0,
        total_training_count - 1,
        min(total_training_count, 200_000),
        dtype=np.int64,
    )
    training_positions = get_training_rows(sample_indices)
    all_y = np.concatenate((training_positions[:, 1], generated[:, 1]))
    y_min = min(-2.0, float(np.quantile(all_y, 0.01)) - 0.2)
    y_max = max(2.0, float(np.quantile(all_y, 0.99)) + 0.2)
    x_edges = np.linspace(x_min, x_max, 101)
    y_edges = np.linspace(y_min, y_max, 81)
    x_line = np.linspace(x_min, x_max, 500 if potential_type == "chain" else 300)
    energy_line = potential(
        x_line,
        np.zeros_like(x_line),
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    generated_plot = generated
    if early_time is None:
        early_time = float(trajectory_times[0])
    if late_time is None:
        late_time = float(trajectory_times[-1])
    if early_samples is None:
        early_samples = _closest_time_slice(
            generated_plot, generated_time, early_time,
        )
    if late_samples is None:
        late_samples = _closest_time_slice(
            generated_plot, generated_time, late_time,
        )
    early_samples = np.asarray(early_samples)
    late_samples = np.asarray(late_samples)
    if early_samples.ndim != 2 or early_samples.shape[1] < 2 or not len(early_samples):
        raise ValueError("early_samples must contain generated x,y positions")
    if late_samples.ndim != 2 or late_samples.shape[1] < 2 or not len(late_samples):
        raise ValueError("late_samples must contain generated x,y positions")
    train_energy = potential(
        training_positions[:, 0],
        training_positions[:, 1],
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    generated_energy = potential(
        generated_plot[:, 0],
        generated_plot[:, 1],
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=160, constrained_layout=True)

    # [0, 0] and [0, 1] show fixed-time generated samples over the PES.
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    potential_grid = potential(
        x_centers[:, None],
        y_centers[None, :],
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    # [0, 0] and [0, 1] show generated samples at selected early/late times.
    for ax, samples, sample_time, title in (
        (axes[0, 0], early_samples, early_time, "Early"),
        (axes[0, 1], late_samples, late_time, "Late"),
    ):
        ax.contourf(
            x_centers,
            y_centers,
            potential_grid.T,
            levels=24,
            cmap="Greys",
            alpha=0.8,
        )
        ax.contour(
            x_centers,
            y_centers,
            potential_grid.T,
            levels=12,
            colors="0.5",
            linewidths=0.6,
        )
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            s=7,
            alpha=0.65,
            color="tab:orange",
        )
        ax.set_title(f"Generated samples at {title.lower()} time (t={sample_time:g})")

    # [0, 2] All generated samples colored by their physical-time conditions.
    generated_color = generated[:, 2] if objective == "joint" else generated_time
    generated_scatter = axes[0, 2].scatter(
        generated_plot[:, 0],
        generated_plot[:, 1],
        c=generated_color,
        cmap="viridis",
        s=5,
        alpha=0.6,
    )
    axes[0, 2].set_title("Generated samples (colored by time)")
    fig.colorbar(
        generated_scatter, ax=axes[0, 2], label="Physical MD time",
    )

    # [1, 0] Potential-energy distribution, reference vs generated.
    axes[1, 0].hist(
        train_energy, bins=50, density=True, color="tab:blue",
        alpha=0.7, label="Training MD",
    )
    axes[1, 0].hist(
        generated_energy, bins=50, density=True, color="tab:orange",
        alpha=0.5, label="Generated",
    )
    axes[1, 0].set_title("Potential energy distribution")
    axes[1, 0].set_xlabel("$V(x,y)$")
    axes[1, 0].set_ylabel("Probability density")
    axes[1, 0].legend()

    # [1, 1] x marginal with potential profile.
    axes[1, 1].hist(
        training_positions[:, 0], bins=60, range=(x_min, x_max),
        density=True, alpha=0.55, color="tab:blue", label="Training MD $p(x)$",
    )
    axes[1, 1].hist(
        generated_plot[:, 0], bins=60, range=(x_min, x_max),
        density=True, alpha=0.45, color="tab:orange", label="Generated $p(x)$",
    )
    potential_axis = axes[1, 1].twinx()
    potential_axis.plot(x_line, energy_line, "k--", linewidth=1.5, label="$V(x,0)$")
    axes[1, 1].set_title("x marginal and potential")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Probability density")
    potential_axis.set_ylabel("Potential energy")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    potential_handles, potential_labels = potential_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + potential_handles, labels + potential_labels, fontsize=7)

    # [1, 2] Learned score/velocity at the requested physical time.
    if field_time is None:
        field_time = float(trajectory_times[0])
    if not np.isfinite(field_time) or not trajectory_times[0] <= field_time <= trajectory_times[-1]:
        raise ValueError(
            f"field_time must be between {trajectory_times[0]:g} and "
            f"{trajectory_times[-1]:g}"
        )
    primary_quiver, primary_magnitude, primary_label = _plot_conditional_3d_field(
        axes[1, 2],
        model,
        model_type,
        field_config,
        potential_type,
        barrier_amplitude,
        barrier_width,
        period,
        well_amplitude,
        x_min,
        x_max,
        y_min,
        y_max,
        field_time,
        field_initial_position,
        show_quiver_key=False,
    )

    # [2, 2] Same field at a second time, defaulting to the archive midpoint.
    if field_time_secondary is None:
        field_time_secondary = float(trajectory_times[len(trajectory_times) // 2])
    if (
        not np.isfinite(field_time_secondary)
        or not trajectory_times[0] <= field_time_secondary <= trajectory_times[-1]
    ):
        raise ValueError(
            f"field_time_secondary must be between {trajectory_times[0]:g} and "
            f"{trajectory_times[-1]:g}"
        )
    secondary_quiver, secondary_magnitude, secondary_label = _plot_conditional_3d_field(
        axes[2, 2],
        model,
        model_type,
        field_config,
        potential_type,
        barrier_amplitude,
        barrier_width,
        period,
        well_amplitude,
        x_min,
        x_max,
        y_min,
        y_max,
        field_time_secondary,
        field_initial_position,
        show_quiver_key=False,
    )
    shared_field_strength = max(primary_magnitude, secondary_magnitude)
    if shared_field_strength > 0:
        quiver_scale = shared_field_strength / 0.4
        for ax, quiver, field_label in (
            (axes[1, 2], primary_quiver, primary_label),
            (axes[2, 2], secondary_quiver, secondary_label),
        ):
            quiver.scale = quiver_scale
            ax.quiverkey(
                quiver,
                0.03,
                0.96,
                shared_field_strength,
                field_label,
                labelpos="E",
                coordinates="axes",
                fontproperties={"size": 7},
            )

    # [2, 0] Generated x samples over physical MD time.
    axes[2, 0].scatter(
        generated_time,
        generated_plot[:, 0],
        s=5,
        alpha=0.5,
        color="tab:orange",
        label="Generated x",
    )
    axes[2, 0].set_title("Generated x versus physical MD time")
    axes[2, 0].set_xlabel("Physical MD time")
    axes[2, 0].set_ylabel("Generated x")
    axes[2, 0].legend(fontsize=7)

    # [2, 1] Generated y samples over physical MD time.
    axes[2, 1].scatter(
        generated_time,
        generated_plot[:, 1],
        s=5,
        alpha=0.5,
        color="tab:green",
        label="Generated y",
    )
    axes[2, 1].set_title("Generated y versus physical MD time")
    axes[2, 1].set_xlabel("Physical MD time")
    axes[2, 1].set_ylabel("Generated y")
    axes[2, 1].legend(fontsize=7)

    for ax in axes.flat:
        if ax is not potential_axis and ax.get_visible():
            ax.set_xlabel(ax.get_xlabel() or "x")
        if ax.get_ylabel() == "":
            ax.set_ylabel("y")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_learned_field(
    output,
    model,
    model_type,
    field_config,
    potential_type,
    barrier_amplitude,
    barrier_width,
    period,
    well_amplitude,
    training_positions,
    physical_time,
    initial_position,
):
    training_positions = np.asarray(training_positions)
    if training_positions.ndim == 3:
        training_positions = training_positions.reshape(-1, training_positions.shape[-1])
    training_positions = training_positions[:, :2]
    if potential_type == "chain":
        x_min, x_max = 0.0, period
    else:
        x_min, x_max = -3.5, 3.5
    y_min = min(-2.0, float(np.quantile(training_positions[:, 1], 0.01)) - 0.2)
    y_max = max(2.0, float(np.quantile(training_positions[:, 1], 0.99)) + 0.2)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160, constrained_layout=True)
    _plot_conditional_3d_field(
        ax,
        model,
        model_type,
        field_config,
        potential_type,
        barrier_amplitude,
        barrier_width,
        period,
        well_amplitude,
        x_min,
        x_max,
        y_min,
        y_max,
        physical_time,
        initial_position,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_conditional_3d_field(
    ax,
    model,
    model_type,
    field_config,
    potential_type,
    barrier_amplitude,
    barrier_width,
    period,
    well_amplitude,
    x_min,
    x_max,
    y_min,
    y_max,
    physical_time,
    initial_position,
    show_quiver_key=True,
):
    x = np.linspace(x_min, x_max, 17)
    y = np.linspace(y_min, y_max, 15)
    X, Y = np.meshgrid(x, y)
    energy = potential(
        X,
        Y,
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    ax.contourf(X, Y, energy, levels=20, cmap="Greys", alpha=0.55)
    if model is None or model_type not in {"diffusion", "flow"} or field_config is None:
        ax.text(
            0.5, 0.5, "Learned field unavailable for this comparison.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("Learned conditional field")
        return None, 0.0, "Learned field"

    condition_dim = int(field_config["condition_dim"])
    condition_mean = np.asarray(field_config.get("condition_mean", []), dtype=np.float32)
    condition_std = np.asarray(field_config.get("condition_std", []), dtype=np.float32)
    state_mean = np.asarray(field_config["state_mean"], dtype=np.float32)
    state_std = np.asarray(field_config["state_std"], dtype=np.float32)
    period_scale = period if potential_type == "chain" else 1.0
    state_points = np.column_stack((X.ravel(), Y.ravel())).astype(np.float32)
    if potential_type == "chain":
        state_points[:, 0] /= period
    model_state_dim = int(field_config["state_dim"])
    if model_state_dim == 3:
        state_points = np.column_stack((
            state_points,
            np.full(len(state_points), physical_time, dtype=np.float32),
        ))
    normalized_states = (state_points - state_mean) / state_std

    condition_tensor = None
    if condition_dim:
        condition_values = np.broadcast_to(
            np.asarray(
                [physical_time, initial_position[0], initial_position[1]],
                dtype=np.float32,
            ),
            (len(state_points), condition_dim),
        ).copy()
        if potential_type == "chain":
            condition_values[:, 1] = np.mod(condition_values[:, 1], period) / period
        normalized_condition = (condition_values - condition_mean) / condition_std
        condition_tensor = torch.as_tensor(
            normalized_condition, dtype=torch.float32, device=DEVICE,
        )

    states = torch.as_tensor(normalized_states, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        if model_type == "diffusion":
            sigma = torch.full((len(states),), SIGMA_MIN, device=DEVICE)
            field = model(states, sigma, condition_tensor).cpu().numpy()
            field = field[:, :2] / (state_std[:2] * period_scale)
            field_label = f"Score ($\\sigma={SIGMA_MIN:g}$)"
        else:
            tau = torch.full((len(states),), 0.5, device=DEVICE)
            field = model(states, tau, condition_tensor).cpu().numpy()
            field = field[:, :2] * (state_std[:2] * period_scale)
            field_label = "Learned velocity ($\\tau=0.5$)"

    arrows = ax.quiver(
        X, Y, field[:, 0].reshape(X.shape), field[:, 1].reshape(Y.shape),
        color="tab:blue", angles="xy", scale_units="xy", alpha=0.85,
        label=field_label,
    )
    field_magnitude = np.hypot(field[:, 0], field[:, 1])
    max_field_magnitude = np.max(field_magnitude)
    field_strength = np.quantile(field_magnitude, 0.9)
    if show_quiver_key and field_strength > 0:
        ax.quiverkey(
            arrows, 0.03, 0.96, max_field_magnitude, field_label, labelpos="E",
            coordinates="axes", fontproperties={"size": 7},
        )
    elif show_quiver_key:
        ax.text(
            0.03, 0.96, f"{field_label}: zero field",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
        )
    ax.set_title(
        f"Learned field at t={physical_time:g}\n"
        f"init=({initial_position[0]:.2g}, {initial_position[1]:.2g})",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="None", color="tab:red",
                markersize=5,
                label=(
                    "Generated initial position "
                    f"({initial_position[0]:.3g}, {initial_position[1]:.3g})"
                ),
            )
        ],
        loc="lower right",
        fontsize=7,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return arrows, field_strength, field_label


def _sample_conditional_time_slice(
    model,
    model_type,
    settings,
    physical_time,
    initial_position,
    potential_type,
    period,
    sample_count=1000,
):
    from .model_3D import sample_diffusion as sample_diffusion_3d
    from .model_3D import sample_flow as sample_flow_3d

    condition = np.asarray(
        [physical_time, initial_position[0], initial_position[1]],
        dtype=np.float32,
    )
    if potential_type == "chain":
        condition[1] = np.mod(condition[1], period) / period
    condition_mean = np.asarray(settings["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(settings["condition_std"], dtype=np.float32)
    normalized_condition = torch.as_tensor(
        np.tile((condition - condition_mean) / condition_std, (sample_count, 1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    if model_type == "diffusion":
        samples = sample_diffusion_3d(
            model,
            sample_count,
            condition=normalized_condition,
            n_levels=settings.get("noise_levels", 12),
            steps_per_level=settings.get("steps_per_level", 50),
        )
    else:
        samples = sample_flow_3d(
            model,
            sample_count,
            condition=normalized_condition,
            ode_steps=settings.get("ode_steps", 50),
        )
    target_mean = np.asarray(settings["target_mean"], dtype=np.float32)
    target_std = np.asarray(settings["target_std"], dtype=np.float32)
    samples = samples.cpu().numpy() * target_std + target_mean
    if potential_type == "chain":
        samples[:, 0] = np.mod(samples[:, 0], 1.0) * period
    return samples


def _closest_time_slice(samples, sample_times, target_time, sample_count=1000):
    selected = np.argsort(np.abs(sample_times - target_time))[:sample_count]
    return samples[selected]


def _plot_learned_field(
    ax,
    model,
    model_type,
    reference_samples,
    mean,
    std,
    potential_type,
    barrier_amplitude,
    barrier_width,
    period,
    well_amplitude,
    x_min,
    x_max,
):
    x = np.linspace(x_min, x_max, 17)
    y = np.linspace(-2.0, 2.0, 15)
    X, Y = np.meshgrid(x, y)
    energy = potential(
        X,
        Y,
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    ax.contourf(X, Y, energy, levels=20, cmap="Greys", alpha=0.65)

    if model is None or mean is None or std is None:
        ax.text(
            0.5,
            0.5,
            "Model and normalization values are required\nto draw the learned field.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    device = next(model.parameters()).device
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device).reshape(1, -1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device).reshape(1, -1)
    physical_points = np.column_stack((X.ravel(), Y.ravel()))
    model_points = physical_points.copy()
    coordinate_factor = np.ones(2, dtype=np.float32)
    if potential_type == "chain":
        model_points[:, 0] /= period
        coordinate_factor[0] = 1.0 / period
    normalized_points = torch.as_tensor(
        (model_points - mean_tensor.cpu().numpy()) / std_tensor.cpu().numpy(),
        dtype=torch.float32,
        device=device,
    )
    reference_model_points = np.asarray(reference_samples, dtype=np.float32).copy()
    if potential_type == "chain":
        reference_model_points[:, 0] /= period
    reference_normalized = (
        reference_model_points - mean_tensor.cpu().numpy()
    ) / std_tensor.cpu().numpy()
    reference_indices = np.linspace(
        0,
        len(reference_normalized) - 1,
        min(len(reference_normalized), 2_000),
        dtype=np.int64,
    )
    reference_subset = reference_normalized[reference_indices]
    points = normalized_points.cpu().numpy()
    sigma = float(SIGMA_MIN)
    tau = 0.5

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        if model_type == "diffusion":
            condition = torch.full(
                (len(normalized_points),), SIGMA_MIN,
                dtype=torch.float32,
                device=device,
            )
        else:
            condition = torch.full(
                (len(normalized_points),), 0.5,
                dtype=torch.float32,
                device=device,
            )
        field = model(normalized_points, condition).cpu().numpy()
    model.train(model_was_training)

    if model_type == "diffusion":
        kBT = 0.4
        delta = 1e-4
        dV_dx = (
            potential(
                X + delta, Y, potential_type,
                barrier_amplitude=barrier_amplitude,
                barrier_width=barrier_width,
                well_amplitude=well_amplitude,
                period=period,
            )
            - potential(
                X - delta, Y, potential_type,
                barrier_amplitude=barrier_amplitude,
                barrier_width=barrier_width,
                well_amplitude=well_amplitude,
                period=period,
            )
        ) / (2 * delta)
        dV_dy = (
            potential(
                X, Y + delta, potential_type,
                barrier_amplitude=barrier_amplitude,
                barrier_width=barrier_width,
                well_amplitude=well_amplitude,
                period=period,
            )
            - potential(
                X, Y - delta, potential_type,
                barrier_amplitude=barrier_amplitude,
                barrier_width=barrier_width,
                well_amplitude=well_amplitude,
                period=period,
            )
        ) / (2 * delta)
        analytical_field = np.column_stack((
            (-dV_dx / kBT).ravel(),
            (-dV_dy / kBT).ravel(),
        ))
        physical_score_scale = (
            std_tensor.cpu().numpy().reshape(-1) / coordinate_factor
        )
        field /= physical_score_scale
        label = f"Learned score ($\\sigma={SIGMA_MIN:g}$)"
        analytical_label = r"Analytical Boltzmann score: $-\nabla V/k_BT$"
    else:
        variance = (1.0 - tau) ** 2
        analytical_field = np.empty_like(points)
        for start in range(0, len(points), 32):
            batch = points[start : start + 32]
            offsets = batch[:, None, :] - tau * reference_subset[None, :, :]
            log_weights = -0.5 * np.sum(offsets**2, axis=2) / variance
            log_weights -= log_weights.max(axis=1, keepdims=True)
            weights = np.exp(log_weights)
            weights /= weights.sum(axis=1, keepdims=True)
            target_velocities = (
                reference_subset[None, :, :] - batch[:, None, :]
            ) / (1.0 - tau)
            analytical_field[start : start + len(batch)] = np.sum(
                weights[..., None] * target_velocities,
                axis=1,
            )
        physical_velocity_scale = (
            std_tensor.cpu().numpy().reshape(-1) / coordinate_factor
        )
        field *= physical_velocity_scale
        analytical_field *= physical_velocity_scale
        label = f"Learned velocity ($\\tau={tau:g}$)"
        analytical_label = (
            r"Analytical flow target (training data): "
            r"$v_\tau^*(x)=\mathbb{E}[(Z-x)/(1-\tau)\mid X_\tau=x]$"
        )

    field_shape = X.shape
    learned_u = field[:, 0].reshape(field_shape)
    learned_v = field[:, 1].reshape(field_shape)
    analytical_u = analytical_field[:, 0].reshape(field_shape)
    analytical_v = analytical_field[:, 1].reshape(field_shape)
    difference = np.linalg.norm(field - analytical_field, axis=1)
    rmse = float(np.sqrt(np.mean(difference**2)))
    difference_plot = ax.scatter(
        X,
        Y,
        c=difference.reshape(field_shape),
        cmap="magma",
        s=14,
        alpha=0.8,
        zorder=2,
    )
    ax.quiver(
        X,
        Y,
        analytical_u,
        analytical_v,
        color="tab:orange",
        alpha=0.85,
        angles="xy",
        scale_units="xy",
        zorder=3,
    )
    ax.quiver(
        X,
        Y,
        learned_u,
        learned_v,
        color="tab:blue",
        alpha=0.8,
        angles="xy",
        scale_units="xy",
        zorder=4,
    )
    fig = ax.figure
    fig.colorbar(difference_plot, ax=ax, label="$|s_\\theta-s^*|$" if model_type == "diffusion" else "$|v_\\theta-v^*|$")
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", label=label),
            Line2D([0], [0], color="tab:orange", label=analytical_label),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="tab:red",
                label=f"Difference magnitude (RMSE={rmse:.3g})",
            ),
        ],
        loc="lower left",
        fontsize=6,
        framealpha=0.9,
    )


def plot_results(
    true_samples,
    generated,
    output_path,
    model_type="diffusion",
    potential_type="double-well",
    barrier_amplitude=0.0,
    barrier_width=0.25,
    show=False,
    period=CHAIN_PERIOD,
    well_amplitude=3.0,
    model=None,
    mean=None,
    std=None,
    md_time=None,
    stop_time=None,
):
    if model_type not in {"diffusion", "flow"}:
        raise ValueError(f"Unsupported model type for plotting: {model_type}")
    if potential_type not in {"double-well", "double-well-Hbarrier", "chain"}:
        raise ValueError(f"Unsupported potential type for plotting: {potential_type}")
    model_label = model_type.title()
    true_samples = np.asarray(true_samples)
    generated = np.asarray(generated)
    if (
        true_samples.ndim != 2 or true_samples.shape[1] < 2
        or generated.ndim != 2 or generated.shape[1] < 2
        or not len(true_samples) or not len(generated)
    ):
        raise ValueError("Plotting requires non-empty 2D reference and generated samples")
    plot_samples = true_samples[::max(1, len(true_samples) // 200_000)]
    x_min, x_max = (0.0, period) if potential_type == "chain" else (-3.5, 3.5)

    fig, axes = plt.subplots(3, 2, figsize=(13, 14), dpi=160)
    x = np.linspace(x_min, x_max, 400 if potential_type == "chain" else 200)
    y = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x, y)
    energy_grid = potential(
        X,
        Y,
        potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=period,
    )
    x_edges = np.linspace(x_min, x_max, 129 if potential_type == "chain" else 71)
    y_edges = np.linspace(-2.0, 2.0, 51)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    density_x, density_y = np.meshgrid(x_centers, y_centers)
    sample_counts, _, _ = np.histogram2d(
        true_samples[:, 0], true_samples[:, 1], bins=(x_edges, y_edges),
    )
    density_plot = axes[0, 0].pcolormesh(
        x_edges, y_edges, sample_counts.T, cmap="plasma", shading="auto", alpha=0.8,
    )
    fig.colorbar(density_plot, ax=axes[0, 0], label="MD sample count")
    axes[0, 0].contour(X, Y, energy_grid, levels=12, cmap="Greys", alpha=0.7)
    axes[0, 0].set_title("Training MD sample density over $V(x,y)$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].contour(X, Y, energy_grid, levels=12, cmap="Greys", alpha=0.25)
    for samples, label, color, linestyle in (
        (plot_samples, "MD", "tab:blue", "solid"),
        (generated, model_label, "tab:orange", "dashed"),
    ):
        counts, _, _ = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=(x_edges, y_edges),
        )
        density = counts.T / (len(samples) * np.diff(x_edges)[0] * np.diff(y_edges)[0])
        peak_density = density.max()
        if peak_density > 0:
            axes[0, 1].contour(
                density_x,
                density_y,
                density,
                levels=peak_density * np.array([0.15, 0.35, 0.6]),
                colors=[color],
                linestyles=linestyle,
                linewidths=1.5,
            )
    axes[0, 1].legend(handles=[
        Line2D([0], [0], color="tab:blue", linestyle="solid", label="MD density"),
        Line2D([0], [0], color="tab:orange", linestyle="dashed", label=f"{model_label} density"),
    ])
    axes[0, 1].set_xlim(x_edges[[0, -1]])
    axes[0, 1].set_ylim(y_edges[[0, -1]])
    axes[0, 1].set_title(
        f"MD vs {model_label} density ({potential_type} potential)",
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[1, 0].hist(
        potential(
            true_samples[:, 0], true_samples[:, 1], potential_type,
            barrier_amplitude=barrier_amplitude, barrier_width=barrier_width,
            well_amplitude=well_amplitude,
            period=period,
        ),
        bins=40,
        density=True,
        alpha=0.65,
        color="tab:blue",
        label="Training MD",
    )
    axes[1, 0].hist(
        potential(
            generated[:, 0], generated[:, 1], potential_type,
            barrier_amplitude=barrier_amplitude, barrier_width=barrier_width,
            well_amplitude=well_amplitude,
            period=period,
        ),
        bins=40,
        density=True,
        alpha=0.5,
        color="tab:orange",
        label=f"{model_label} predictions",
    )
    axes[1, 0].set_title(
        f"{potential_type} potential energy: MD vs {model_label}",
    )
    axes[1, 0].set_xlabel("Potential energy $V(x,y)$")
    axes[1, 0].set_ylabel("Probability density")
    axes[1, 0].legend()
    # [1, 1]: position marginal and potential profile.
    ax3 = axes[1, 1]
    ax3_twin = ax3.twinx()
    x_bins = 128 if potential_type == "chain" else 60
    ax3.hist(
        true_samples[:, 0], bins=x_bins, range=(x_min, x_max), density=True,
        color="tab:blue", alpha=0.45, label="MD p(x)",
    )
    ax3.hist(
        generated[:, 0], bins=x_bins, range=(x_min, x_max), density=True,
        color="tab:orange", alpha=0.55, label=f"{model_label} p(x)",
    )
    x_line = np.linspace(x_min, x_max, 1000 if potential_type == "chain" else 300)
    ax3_twin.plot(
        x_line,
        potential(
            x_line, 0.0, potential_type,
            barrier_amplitude=barrier_amplitude, barrier_width=barrier_width,
            well_amplitude=well_amplitude,
            period=period,
        ),
        color="black", lw=2, linestyle="--", label="V(x,0)",
    )
    ax3.set_title(f"x Marginal & {potential_type} Potential")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Probability Density", color="teal")
    ax3_twin.set_ylabel("Potential Energy V(x,0)", color="black")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.set_xlim(x_min, x_max)

    # [2, 0]: learned score/velocity field over the potential.
    _plot_learned_field(
        axes[2, 0], model, model_type, true_samples, mean, std, potential_type,
        barrier_amplitude, barrier_width, period, well_amplitude, x_min, x_max,
    )
    axes[2, 0].set_title(
        "Learned score field" if model_type == "diffusion" else "Learned velocity field",
    )
    axes[2, 0].set_xlabel("x")
    axes[2, 0].set_ylabel("y")
    axes[2, 0].set_xlim(x_min, x_max)
    axes[2, 0].set_ylim(-2.0, 2.0)

    # [2, 1]: selected training trajectory from burn-in to the requested time.
    md_time = np.arange(len(true_samples), dtype=float) if md_time is None else np.asarray(md_time)
    if md_time.ndim != 1 or len(md_time) != len(true_samples):
        raise ValueError("md_time must match the training reference samples")
    if not np.isfinite(md_time).all() or np.any(np.diff(md_time) < 0):
        raise ValueError("md_time must contain finite, ordered times")
    if stop_time is None:
        stop_time = float(md_time[-1])
    if not np.isfinite(stop_time) or stop_time < md_time[0] or stop_time > md_time[-1]:
        raise ValueError(
            f"stop_time must be between {md_time[0]:g} and {md_time[-1]:g}"
        )
    time_mask = md_time <= stop_time
    plot_times = md_time[time_mask]
    plot_positions = true_samples[time_mask, :2]
    time_stride = max(1, (len(plot_times) + 199_999) // 200_000)
    axes[2, 1].plot(
        plot_times[::time_stride], plot_positions[::time_stride, 0],
        label="Training MD x(t)", color="indigo", alpha=0.85, linewidth=0.8,
    )
    axes[2, 1].plot(
        plot_times[::time_stride], plot_positions[::time_stride, 1],
        label="Training MD y(t)", color="mediumseagreen", alpha=0.75, linewidth=0.8,
    )
    axes[2, 1].set_title(
        f"Training trajectory from burn-in (t={md_time[0]:g}) "
        f"to t={stop_time:g}"
    )
    axes[2, 1].set_xlabel("Physical MD time")
    axes[2, 1].set_ylabel("Position")
    axes[2, 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_checkpoint(
    checkpoint_path,
    plot="loss",
    zoom=False,
    stop_time=None,
    trajectory_count=1,
    loss_curves="both",
    field_time=None,
    field_time_secondary=None,
    early_time=None,
    late_time=None,
    time_slice_samples=1000,
):
    from .checkpoint import (
        checkpoint_settings,
        load_training_checkpoint,
        resolve_checkpoint_data,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    settings = checkpoint_settings(checkpoint_path, checkpoint)
    run_name = checkpoint_path.stem.removesuffix("_checkpoint")
    output_dir = checkpoint_path.parent

    if plot in {"loss", "both"}:
        training_history = checkpoint.get("training_history")
        validation_history = checkpoint.get("validation_history")
        if loss_curves in {"both", "training"} and not training_history:
            raise ValueError(f"Checkpoint has no training loss history: {checkpoint_path}")
        if loss_curves in {"both", "validation"} and not validation_history:
            raise ValueError(f"Checkpoint has no validation loss history: {checkpoint_path}")
        loss_output = output_dir / f"{run_name}_loss.png"
        plot_training_loss(
            training_history or [],
            loss_output,
            validation_history,
            show=zoom,
            loss_curves=loss_curves,
        )
        print(f"Training loss saved to {loss_output}")

    if plot not in {"results", "both", "field"}:
        return

    data_path = resolve_checkpoint_data(
        checkpoint_path, settings, Path(__file__).resolve().parents[1],
    )
    generated_path = output_dir / f"{run_name}_generated.npz"
    objective = settings.get("objective")
    if plot == "field" and objective is None:
        raise ValueError("--plot field is only available for 3D checkpoints")
    if objective is not None:
        from .data import evenly_spaced_indices
        from .model_3D import (
            Diffusion3D,
            Flow3D,
            sample_diffusion as sample_diffusion_3d,
            sample_flow as sample_flow_3d,
        )

        with np.load(data_path) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float32)
            times = np.asarray(archive["md_time"], dtype=np.float32)
            initial_positions = (
                np.asarray(archive["initial_positions"], dtype=np.float32)
                if "initial_positions" in archive else None
            )
        if positions.ndim == 2:
            positions = positions[None, ...]
        if positions.ndim != 3 or positions.shape[2] < 2:
            raise ValueError(f"Expected 2D or 3D position data in {data_path}")
        if times.ndim != 1 or len(times) != positions.shape[1]:
            raise ValueError(f"MD times do not match position frames in {data_path}")
        trajectories = settings.get("trajectory", 1)
        trajectories = [trajectories] if isinstance(trajectories, int) else list(trajectories)
        trajectory_indices = np.asarray(trajectories, dtype=np.int64) - 1
        if (
            len(trajectory_indices) == 0
            or np.any(trajectory_indices < 0)
            or np.any(trajectory_indices >= len(positions))
        ):
            raise ValueError(f"Checkpoint trajectory is unavailable in {data_path}")
        if trajectory_count < 1 or trajectory_count > len(trajectory_indices):
            raise ValueError(
                f"trajectory_count must be between 1 and {len(trajectory_indices)} "
                "selected checkpoint trajectories"
            )
        trajectory_indices = trajectory_indices[:trajectory_count]
        reference_positions = positions[trajectory_indices, :, :2]
        if settings.get("potential_type") == "chain":
            reference_positions = reference_positions.copy()
            reference_positions[:, 0] %= settings.get("period", CHAIN_PERIOD)
        reference_time = times
        reference = (
            np.column_stack((reference_positions[0], times))
            if objective == "joint"
            else reference_positions.reshape(-1, 2)
        )
        model_name = settings.get("model")
        model_class = {"diffusion": Diffusion3D, "flow": Flow3D}.get(model_name)
        if model_class is None:
            raise ValueError(f"Checkpoint has unsupported model type: {model_name}")
        state_dim = int(settings.get("state_dim", 0))
        condition_dim = int(settings.get("condition_dim", 0))
        if objective not in {"joint", "condition", "AdaLN"}:
            raise ValueError(f"Checkpoint has unsupported 3D objective: {objective}")
        is_conditional = objective in {"condition", "AdaLN"}
        if state_dim != (2 if is_conditional else 3) or condition_dim != (
            3 if is_conditional else 0
        ):
            raise ValueError(
                f"Checkpoint dimensions do not match objective {objective}: "
                f"state_dim={state_dim}, condition_dim={condition_dim}"
            )
        model = model_class(
            state_dim,
            condition_dim,
            adaln=objective == "AdaLN",
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        if is_conditional:
            field_config = {
                "state_dim": state_dim,
                "condition_dim": condition_dim,
                "state_mean": settings["target_mean"],
                "state_std": settings["target_std"],
                "condition_mean": settings["condition_mean"],
                "condition_std": settings["condition_std"],
            }
        else:
            field_config = {
                "state_dim": state_dim,
                "condition_dim": condition_dim,
                "state_mean": settings["mean"],
                "state_std": settings["std"],
            }
        from .data import get_potential_settings
        (
            potential_type,
            barrier_amplitude,
            barrier_width,
            well_amplitude,
        ) = get_potential_settings(data_path)
        period = settings.get("period", CHAIN_PERIOD)
        early_time = float(times[0]) if early_time is None else float(early_time)
        late_time = float(times[-1]) if late_time is None else float(late_time)
        for label, selected_time in (
            ("early_time", early_time),
            ("late_time", late_time),
        ):
            if not np.isfinite(selected_time) or not times[0] <= selected_time <= times[-1]:
                raise ValueError(
                    f"{label} must be between {times[0]:g} and {times[-1]:g}"
                )
        if plot == "field":
            selected_field_time = (
                float(times[0]) if field_time is None else float(field_time)
            )
            if not np.isfinite(selected_field_time) or not times[0] <= selected_field_time <= times[-1]:
                raise ValueError(
                    f"field_time must be between {times[0]:g} and {times[-1]:g}"
                )
            if is_conditional:
                if initial_positions is None:
                    raise ValueError(
                        f"Checkpoint objective {objective} requires initial_positions in {data_path}"
                    )
                if (
                    initial_positions.shape != (len(positions), 2)
                    or not np.isfinite(initial_positions).all()
                ):
                    raise ValueError(f"Invalid initial_positions in {data_path}")
                field_initial_position = initial_positions[trajectory_indices[0]]
            else:
                field_initial_position = reference_positions[0, 0]
            field_output = output_dir / f"{run_name}_field.png"
            plot_learned_field(
                field_output,
                model,
                model_name,
                field_config,
                potential_type,
                barrier_amplitude,
                barrier_width,
                period,
                well_amplitude,
                positions[..., :2],
                selected_field_time,
                field_initial_position,
            )
            print(f"Learned field plot saved to {field_output}")
            return
        if generated_path.exists():
            with np.load(generated_path) as data:
                generated = data["generated"]
                generated_time = data["generated_time"]
                cached_objective = str(data["objective"].item())
                if cached_objective != objective:
                    raise ValueError(
                        f"Cached samples use objective {cached_objective}, "
                        f"but checkpoint uses {objective}"
                    )
        else:
            sample_count = int(settings.get("generated_samples", 1_000))

            if is_conditional:
                if initial_positions is None:
                    raise ValueError(
                        f"Checkpoint objective {objective} requires initial_positions in {data_path}"
                    )
                if (
                    initial_positions.shape != (len(positions), 2)
                    or not np.isfinite(initial_positions).all()
                ):
                    raise ValueError(f"Invalid initial_positions in {data_path}")
                selected_initial_positions = initial_positions[trajectory_indices]
                times_flat = np.broadcast_to(
                    times[None, :], (len(trajectory_indices), len(times)),
                ).reshape(-1)
                initial_positions_flat = np.broadcast_to(
                    selected_initial_positions[:, None, :],
                    (len(trajectory_indices), len(times), 2),
                ).reshape(-1, 2)
                condition_raw = np.column_stack((
                    times_flat, initial_positions_flat,
                )).astype(np.float32)
                if settings.get("potential_type") == "chain":
                    period = settings.get("period", CHAIN_PERIOD)
                    condition_raw[:, 1] = np.mod(condition_raw[:, 1], period) / period
                sample_indices = evenly_spaced_indices(
                    len(condition_raw),
                    min(int(settings.get("max_samples", 100_000)), len(condition_raw)),
                )
                condition_mean = torch.as_tensor(
                    settings["condition_mean"], dtype=torch.float32, device=DEVICE,
                )
                condition_std = torch.as_tensor(
                    settings["condition_std"], dtype=torch.float32, device=DEVICE,
                )
                conditions = torch.as_tensor(
                    (condition_raw[sample_indices] - condition_mean.cpu().numpy())
                    / condition_std.cpu().numpy(),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                condition_indices = torch.randint(
                    len(conditions), (sample_count,), device=DEVICE,
                )
                sampled_conditions = conditions[condition_indices]
                if model_name == "diffusion":
                    generated_normalized = sample_diffusion_3d(
                        model,
                        sample_count,
                        condition=sampled_conditions,
                        n_levels=settings.get("noise_levels", 12),
                        steps_per_level=settings.get("steps_per_level", 50),
                    )
                else:
                    generated_normalized = sample_flow_3d(
                        model,
                        sample_count,
                        condition=sampled_conditions,
                        ode_steps=settings.get("ode_steps", 50),
                    )
                target_mean = torch.as_tensor(
                    settings["target_mean"], dtype=torch.float32, device=DEVICE,
                )
                target_std = torch.as_tensor(
                    settings["target_std"], dtype=torch.float32, device=DEVICE,
                )
                generated = (
                    generated_normalized.cpu().numpy() * target_std.cpu().numpy()
                    + target_mean.cpu().numpy()
                )
                generated_conditions = condition_raw[
                    sample_indices[condition_indices.cpu().numpy()]
                ]
                generated_time = generated_conditions[:, 0]
                if settings.get("potential_type") == "chain":
                    generated[:, 0] = np.mod(generated[:, 0], 1.0) * settings.get(
                        "period", CHAIN_PERIOD,
                    )
            else:
                mean = torch.as_tensor(settings["mean"], dtype=torch.float32, device=DEVICE)
                std = torch.as_tensor(settings["std"], dtype=torch.float32, device=DEVICE)
                if model_name == "diffusion":
                    generated_normalized = sample_diffusion_3d(
                        model,
                        sample_count,
                        n_levels=settings.get("noise_levels", 12),
                        steps_per_level=settings.get("steps_per_level", 50),
                    )
                else:
                    generated_normalized = sample_flow_3d(
                        model, sample_count, ode_steps=settings.get("ode_steps", 50),
                    )
                generated = generated_normalized.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()
                generated_time = generated[:, 2]
                if settings.get("potential_type") == "chain":
                    generated[:, 0] = np.mod(generated[:, 0], 1.0) * settings.get(
                        "period", CHAIN_PERIOD,
                    )

            np.savez_compressed(
                generated_path,
                generated=generated,
                generated_time=generated_time,
                objective=objective,
                model=model_name,
            )
        result_output = output_dir / f"{run_name}_samples.png"
        if objective in {"condition", "AdaLN"}:
            reference_time = np.broadcast_to(
                times[None, :], (len(trajectory_indices), len(times)),
            ).reshape(-1)
        early_samples = late_samples = None
        field_initial_position = None
        if is_conditional:
            if initial_positions is None:
                raise ValueError(
                    f"Checkpoint objective {objective} requires initial_positions in {data_path}"
                )
            if (
                initial_positions.shape != (len(positions), 2)
                or not np.isfinite(initial_positions).all()
            ):
                raise ValueError(f"Invalid initial_positions in {data_path}")
            field_initial_position = initial_positions[trajectory_indices[0]]
            early_samples = _sample_conditional_time_slice(
                model,
                model_name,
                settings,
                early_time,
                field_initial_position,
                potential_type,
                period,
                time_slice_samples,
            )
            late_samples = _sample_conditional_time_slice(
                model,
                model_name,
                settings,
                late_time,
                field_initial_position,
                potential_type,
                period,
                time_slice_samples,
            )
        if is_conditional:
            if initial_positions is None:
                raise ValueError(
                    f"Checkpoint objective {objective} requires initial_positions in {data_path}"
                )
            if (
                initial_positions.shape != (len(positions), 2)
                or not np.isfinite(initial_positions).all()
            ):
                raise ValueError(f"Invalid initial_positions in {data_path}")
        plot_trajectory_comparison(
            reference, generated, reference_time, generated_time,
            objective,
            result_output,
            stop_time=stop_time,
            trajectory_positions=positions[trajectory_indices, :, :2],
            trajectory_times=times,
            model=model,
            model_type=model_name,
            field_config=field_config,
            potential_type=potential_type,
            barrier_amplitude=barrier_amplitude,
            barrier_width=barrier_width,
            period=period,
            well_amplitude=well_amplitude,
            training_positions=positions,
            field_time=field_time,
            field_time_secondary=field_time_secondary,
            early_samples=early_samples,
            early_time=early_time,
            late_samples=late_samples,
            late_time=late_time,
            field_initial_position=field_initial_position,
        )
    else:
        from .data import get_potential_settings, load_equilibrium_bundle
        from .model_2D import DiffusionModel, FlowModel

        model_name = settings.get("model")
        model_class = {"diffusion": DiffusionModel, "flow": FlowModel}.get(model_name)
        if model_class is None:
            raise ValueError(f"Checkpoint has unsupported model type: {model_name}")
        model = model_class().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = torch.as_tensor(settings["mean"], dtype=torch.float32, device=DEVICE)
        std = torch.as_tensor(settings["std"], dtype=torch.float32, device=DEVICE)
        if generated_path.exists():
            with np.load(generated_path) as data:
                generated = data["generated"]
        elif model_name == "diffusion":
            generated = sample_score(
                model,
                mean,
                std,
                n_samples=settings.get("generated_samples", 1_000),
                n_levels=settings.get("noise_levels", 12),
                steps_per_level=settings.get("steps_per_level", 50),
            )
        else:
            generated = sample_flow(
                model,
                mean,
                std,
                n_samples=settings.get("generated_samples", 1_000),
                ode_steps=settings.get("ode_steps", 50),
            )

        true_samples, _, _, _ = load_equilibrium_bundle(
            data_path,
            settings.get("max_samples", 100_000),
            settings.get("diffusion_frames", 500_000),
            trajectory=settings.get("trajectory", 1),
        )
        true_samples = true_samples[:, :2]
        with np.load(data_path) as data:
            training_md_time = np.asarray(data["md_time"], dtype=np.float64)
        potential_type, barrier_amplitude, barrier_width, well_amplitude = (
            get_potential_settings(data_path)
        )
        period = settings.get("period", CHAIN_PERIOD)
        if potential_type == "chain":
            true_samples[:, 0] %= period
            generated[:, 0] %= period
        result_output = output_dir / f"{run_name}_samples.png"
        plot_results(
            true_samples,
            generated,
            result_output,
            model_type=model_name,
            potential_type=potential_type,
            barrier_amplitude=barrier_amplitude,
            barrier_width=barrier_width,
            period=period,
            well_amplitude=well_amplitude,
            model=model,
            mean=mean,
            std=std,
            md_time=training_md_time,
            stop_time=stop_time,
        )
        if not generated_path.exists():
            np.savez_compressed(generated_path, generated=generated)
    print(f"Prediction plot saved to {result_output}")


def main():
    parser = argparse.ArgumentParser(description="Replot prediction outputs from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--plot",
        choices=("loss", "results", "both", "field"),
        default="loss",
        help="Plot loss, full results, both, or only the 3D learned field",
    )
    parser.add_argument(
        "--loss-curves",
        choices=("both", "training", "validation"),
        default="both",
        help="Loss history curves to show when plotting checkpoint loss (default: both)",
    )
    parser.add_argument("--step", type=int, default=1, help="Checkpoint MD time step to load (default: 1)")
    parser.add_argument(
        "--stop-time",
        type=float,
        help="Last physical MD time to show in trajectory panels (default: end of archive)",
    )
    parser.add_argument(
        "--field-time",
        type=float,
        help="Physical MD time at which to evaluate the learned field (default: first saved frame)",
    )
    parser.add_argument(
        "--field-time-secondary",
        type=float,
        help="Physical MD time for the second learned-field panel (default: middle saved frame)",
    )
    parser.add_argument(
        "--early-time",
        type=float,
        help="Physical MD time for generated samples in panel [0,0] (default: first saved frame)",
    )
    parser.add_argument(
        "--late-time",
        type=float,
        help="Physical MD time for generated samples in panel [0,1] (default: last saved frame)",
    )
    parser.add_argument(
        "--time-slice-samples",
        type=int,
        default=1000,
        help="Number of generated samples in each fixed-time panel (default: 1000)",
    )
    parser.add_argument(
        "--trajectory-count",
        type=int,
        default=1,
        help="Number of first MD trajectories to include in the time-colored x-y panel (default: 1)",
    )
    parser.add_argument(
        "--zoom", action="store_true",
        help="Open the loss plot interactively so it can be zoomed (default: save PNG only)",
    )
    args = parser.parse_args()
    if args.zoom and args.plot not in {"loss", "both"}:
        parser.error("--zoom requires --plot loss or --plot both")
    if args.stop_time is not None and args.plot in {"loss", "field"}:
        parser.error("--stop-time requires --plot results or --plot both")
    if args.field_time is not None and args.plot not in {"results", "both", "field"}:
        parser.error("--field-time requires --plot results, --plot both, or --plot field")
    if args.field_time_secondary is not None and args.plot not in {"results", "both"}:
        parser.error("--field-time-secondary requires --plot results or --plot both")
    if (
        (args.early_time is not None or args.late_time is not None)
        and args.plot not in {"results", "both"}
    ):
        parser.error("--early-time and --late-time require --plot results or --plot both")
    if args.time_slice_samples < 1:
        parser.error("--time-slice-samples must be at least 1")
    if args.trajectory_count < 1:
        parser.error("--trajectory-count must be at least 1")
    if args.trajectory_count != 1 and args.plot == "loss":
        parser.error("--trajectory-count requires --plot results or --plot both")
    plot_checkpoint(
        args.checkpoint,
        args.plot,
        args.zoom,
        args.stop_time,
        args.trajectory_count,
        args.loss_curves,
        args.field_time,
        args.field_time_secondary,
        args.early_time,
        args.late_time,
        args.time_slice_samples,
    )


if __name__ == "__main__":
    main()
