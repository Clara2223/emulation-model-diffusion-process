import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .data import STATE_DIM
from .model_2D import DEVICE


CHAIN_PERIOD = 32.0         #default 


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


def potential(
    x,
    y,
    potential_type="double-well",
    barrier_amplitude=0.0,
    barrier_width=0.25,
    period=CHAIN_PERIOD,
    well_amplitude=3.0,
):
    if potential_type == "chain":
        centers = (
            period / 8 * np.arange(8)[:, None] + np.array([-1.0, 1.0])
        ).ravel()
        half_period = period / 2
        dx = (np.asarray(x)[..., None] - centers + half_period) % period - half_period
        dy = np.asarray(y)[..., None]
        wells = 3.0 * np.exp(-0.5 * ((dx / 0.5) ** 2 + (dy / 0.5) ** 2))
        return -np.sum(wells, axis=-1) + 0.1 * np.asarray(y) ** 2
    if potential_type not in {"double-well", "double-well-Hbarrier"}:
        raise ValueError(f"Unsupported potential type: {potential_type}")
    g1 = well_amplitude * np.exp(-((x + 1.0) ** 2 + y**2) / (2 * 0.5**2))
    g2 = well_amplitude * np.exp(-((x - 1.0) ** 2 + y**2) / (2 * 0.5**2))
    ridge = 0.0
    if potential_type == "double-well-Hbarrier":
        if barrier_amplitude <= 0 or barrier_width <= 0:
            raise ValueError("H-barrier plots require positive amplitude and width metadata")
        ridge = barrier_amplitude * np.exp(-0.5 * (np.asarray(x) / barrier_width) ** 2)
    return -(g1 + g2) + 0.1 * (x**2 + y**2) + ridge

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
def plot_training_loss(losses, output_path, validation_losses=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, marker="o", linewidth=1.5, label="Training loss")
    if validation_losses is not None:
        validation_epochs = np.arange(1, len(validation_losses) + 1)
        ax.plot(
            validation_epochs, validation_losses, marker="o", linewidth=1.5,
            label="Validation loss",
        )
    ax.set_title("Training Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


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
):
    if model_type not in {"diffusion", "flow"}:
        raise ValueError(f"Unsupported model type for plotting: {model_type}")
    if potential_type not in {"double-well", "double-well-Hbarrier", "chain"}:
        raise ValueError(f"Unsupported potential type for plotting: {potential_type}")
    model_label = model_type.title()
    plot_samples = true_samples[::max(1, len(true_samples) // 200_000)]
    x_min, x_max = (0.0, period) if potential_type == "chain" else (-3.5, 3.5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    x = np.linspace(x_min, x_max, 400 if potential_type == "chain" else 200)
    y = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x, y)
    axes[0, 0].contour(
        X,
        Y,
        potential(
            X, Y, potential_type, barrier_amplitude=barrier_amplitude,
            barrier_width=barrier_width, well_amplitude=well_amplitude, period=period,
        ),
        levels=12,
        cmap="Greys",
        alpha=0.5,
    )
    x_edges = np.linspace(x_min, x_max, 129 if potential_type == "chain" else 71)
    y_edges = np.linspace(-2.0, 2.0, 51)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    density_x, density_y = np.meshgrid(x_centers, y_centers)
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
            axes[0, 0].contour(
                density_x,
                density_y,
                density,
                levels=peak_density * np.array([0.15, 0.35, 0.6]),
                colors=[color],
                linestyles=linestyle,
                linewidths=1.5,
            )
    axes[0, 0].legend(handles=[
        Line2D([0], [0], color="tab:blue", linestyle="solid", label="MD density"),
        Line2D([0], [0], color="tab:orange", linestyle="dashed", label=f"{model_label} density"),
    ])
    axes[0, 0].set_xlim(x_edges[[0, -1]])
    axes[0, 0].set_ylim(y_edges[[0, -1]])
    axes[0, 0].set_title(
        f"MD vs {model_label} density ({potential_type} potential)",
    )
    axes[0, 1].hist(
        potential(
            plot_samples[:, 0], plot_samples[:, 1], potential_type,
            barrier_amplitude=barrier_amplitude, barrier_width=barrier_width,
            well_amplitude=well_amplitude,
            period=period,
        ),
        bins=40,
        density=True,
        alpha=0.5,
        label="MD",
    )
    axes[0, 1].hist(
        potential(
            generated[:, 0], generated[:, 1], potential_type,
            barrier_amplitude=barrier_amplitude, barrier_width=barrier_width,
            well_amplitude=well_amplitude,
            period=period,
        ),
        bins=40,
        density=True,
        alpha=0.7,
        label=model_label,
    )
    axes[0, 1].set_title(
        f"{potential_type} potential energy: MD vs {model_label}",
    )
    axes[0, 1].legend()
    '''axes[1, 0].plot(generated[:, 0], linewidth=0.7)
    axes[1, 0].set_title("Generated x samples")
    axes[1, 1].plot(generated[:, 1], linewidth=0.7)
    axes[1, 1].set_title("Generated y samples")'''
    
    # [1, 0] SUBFIGUR: 1D Position Marginal vs Potential Profile ---
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    # Histogram af x-værdier (husk at definere x_traj_arr før, f.eks. true_samples[:, 0] eller generated[:, 0])
    # Her antager jeg, at du vil plotte generated x-samples:
    x_bins = 128 if potential_type == "chain" else 60
    ax3.hist(
        plot_samples[:, 0], bins=x_bins, range=(x_min, x_max), density=True,
        color="tab:blue", alpha=0.45, label="MD p(x)",
    )
    ax3.hist(
        generated[:, 0], bins=x_bins, range=(x_min, x_max), density=True,
        color="tab:orange", alpha=0.55, label=f"{model_label} p(x)",
    )
    
    # Potentiale kurve på tvilling-aksen
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
    ax3.set_ylabel("Probability Density", color='teal')
    ax3_twin.set_ylabel("Potential Energy V(x,0)", color='black')
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.set_xlim(x_min, x_max)
    
    # Kombineret plot for Generated X & Y Samples ---
    axes[1, 1].plot(generated[:, 0], linewidth=0.7, label='Generated x', alpha=0.8)
    axes[1, 1].plot(generated[:, 1], linewidth=0.7, label='Generated y', alpha=0.8)
    axes[1, 1].set_title("Generated X & Y Samples")
    axes[1, 1].set_xlabel("Steps / Index")
    axes[1, 1].set_ylabel("Position")
    axes[1, 1].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
