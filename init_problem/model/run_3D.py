import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import evenly_spaced_indices, get_potential_settings
from .checkpoint import (
    checkpoint_option,
    checkpoint_settings,
    load_training_checkpoint,
    resolve_checkpoint_data,
)
from .model_3D import (
    Diffusion3D,
    Flow3D,
    sample_diffusion,
    sample_flow,
    train_diffusion,
    train_flow,
)
from .model_2D import DEVICE
from .predict import histogram_distances, log_prediction, plot_training_loss


def _crossing_rate(positions, times, potential_type, period):
    if potential_type == "chain":
        bin_width = period / 16
        labels = np.floor(np.mod(positions[..., 0], period) / bin_width).astype(np.int64)
    else:
        labels = positions[..., 0] >= 0.0
    if positions.ndim == 2:
        labels = labels[None, :]
        times = np.asarray(times)[None, :]
    durations = np.asarray(times)[:, -1] - np.asarray(times)[:, 0]
    if np.any(durations <= 0):
        raise ValueError("Each trajectory must span positive physical MD time")
    crossings = np.count_nonzero(labels[:, 1:] != labels[:, :-1], axis=1)
    return float(crossings.sum() / durations.sum())


def _save_comparison(reference, generated, reference_time, generated_time, objective, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    if objective == "joint":
        for ax, samples, title in (
            (axes[0], reference, "MD reference"),
            (axes[1], generated, "Generated joint samples"),
        ):
            points = ax.scatter(samples[:, 0], samples[:, 1], c=samples[:, 2], s=3, alpha=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
        fig.colorbar(points, ax=axes, label="Physical MD time")
    else:
        count = min(len(reference), 20_000)
        axes[0].plot(reference_time[:count], reference[:count, 0], linewidth=0.6)
        axes[0].set_title("MD reference trajectory")
        axes[0].set_xlabel("Physical MD time")
        axes[0].set_ylabel("x")
        for index in range(min(len(generated), 8)):
            axes[1].plot(
                generated_time[index], generated[index, :, 0], linewidth=0.7, alpha=0.8,
            )
        axes[1].set_title("Generated trajectories")
        axes[1].set_xlabel("Physical MD time")
        axes[1].set_ylabel("x")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


#paths
### double well
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_1e8steps_5e4burn_20261021_2056.npz"
#### chain of wells
DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_chain_1e8steps_5e4burn_20261021_2133.npz"
### high barrier double well
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_high_barrier_double_well_1e8steps_5e4burn_20261021_time.npz"


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D (x, y, physical MD time) equilibrium or transition models"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--period", type=float)
    parser.add_argument("--objective", choices=("joint", "transitions"))
    parser.add_argument("--model", choices=("diffusion", "flow"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--trajectory-length", type=int, default=100)
    parser.add_argument("--n-levels", type=int)
    parser.add_argument(
        "--steps-per-level", type=int,
        help="Score-based Langevin updates at each noise level (diffusion)",
    )
    parser.add_argument(
        "--ode-steps", type=int,
        help="Numerical integration steps over the flow transport interval",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save model and optimizer state after training",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    resume_checkpoint = None
    settings = {}
    if args.resume_checkpoint is not None:
        resume_checkpoint = load_training_checkpoint(args.resume_checkpoint)
        settings = checkpoint_settings(args.resume_checkpoint, resume_checkpoint)
        if args.data is None:
            args.data = resolve_checkpoint_data(
                args.resume_checkpoint, settings, Path(__file__).resolve().parents[1],
            )
    if args.data is None:
        parser.error("--data is required unless --resume-checkpoint is used")
    args.period = checkpoint_option(parser, args.period, settings, "period", 32.0)
    args.objective = checkpoint_option(
        parser, args.objective, settings, "objective", "joint",
    )
    args.model = checkpoint_option(parser, args.model, settings, "model", "diffusion")
    args.max_samples = checkpoint_option(
        parser, args.max_samples, settings, "max_samples", 100_000, "samples",
    )
    args.n_levels = checkpoint_option(
        parser, args.n_levels, settings, "n_levels", 12, "noise_levels",
    )
    args.steps_per_level = checkpoint_option(
        parser, args.steps_per_level, settings, "steps_per_level", 50,
    )
    args.ode_steps = checkpoint_option(parser, args.ode_steps, settings, "ode_steps", 50)
    if min(args.epochs, args.max_samples, args.generated_samples,
           args.n_levels, args.steps_per_level, args.ode_steps) < 1:
        parser.error("training, sampling, and data-count arguments must be positive")
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2 for train/validation splits")
    if args.period <= 0:
        parser.error("--period must be positive")
    if min(args.trajectories, args.trajectory_length) < 2:
        parser.error("--trajectories and --trajectory-length must be at least 2")

    with np.load(args.data) as archive:
        positions = np.asarray(archive["positions"], dtype=np.float32)
        times = np.asarray(archive["md_time"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] < 2:
        parser.error("The archive must contain positions with x and y columns")
    if len(positions) != len(times):
        parser.error("The archive's positions and md_time arrays must have equal lengths")
    if len(times) < 3 or not np.isfinite(positions[:, :2]).all() or not np.isfinite(times).all():
        parser.error("The archive needs at least three finite position/time samples")
    potential_type, _, _, _ = get_potential_settings(args.data)
    reference_positions = positions[:, :2].copy()
    if potential_type == "chain":
        reference_positions[:, 0] %= args.period
    rng = np.random.default_rng(7)

    if args.objective == "joint":
        reference = np.column_stack((reference_positions, times)).astype(np.float32)
        model_reference = reference.copy()
        if potential_type == "chain":
            model_reference[:, 0] = np.mod(model_reference[:, 0], args.period) / args.period
        count = min(args.max_samples, len(reference))
        selected = evenly_spaced_indices(len(reference), count)
        samples = torch.from_numpy(model_reference[selected])
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(7))
        validation_count = max(1, int(count * 0.1))
        train_indices, val_indices = permutation[validation_count:], permutation[:validation_count]
        training = samples[train_indices]
        validation = samples[val_indices]
        mean = training.mean(dim=0)
        std = training.std(dim=0).clamp_min(1e-6)
        train_states = ((training - mean) / std).to(DEVICE)
        val_states = ((validation - mean) / std).to(DEVICE)
        train_conditions = val_conditions = None
        model_state_dim, condition_dim = 3, 0
        reference_for_metrics = reference
    else:
        if args.trajectory_length > len(times):
            parser.error("--trajectory-length cannot exceed the saved trajectory length")
        pairs = min(args.max_samples, len(positions) - 1)
        selected = evenly_spaced_indices(len(positions) - 1, pairs)
        conditions = np.column_stack(
            (positions[selected, :2], times[selected]),
        ).astype(np.float32)
        targets = positions[selected + 1, :2].astype(np.float32)
        if potential_type == "chain":
            conditions[:, 0] = np.mod(conditions[:, 0], args.period) / args.period
            targets[:, 0] = np.mod(targets[:, 0], args.period) / args.period
        permutation = torch.randperm(pairs, generator=torch.Generator().manual_seed(7))
        validation_count = max(1, int(pairs * 0.1))
        train_indices = permutation[validation_count:].numpy()
        val_indices = permutation[:validation_count].numpy()
        train_condition_raw = conditions[train_indices]
        val_condition_raw = conditions[val_indices]
        train_target_raw = targets[train_indices]
        val_target_raw = targets[val_indices]
        condition_mean = train_condition_raw.mean(axis=0)
        condition_std = np.maximum(train_condition_raw.std(axis=0), 1e-6)
        target_mean = train_target_raw.mean(axis=0)
        target_std = np.maximum(train_target_raw.std(axis=0), 1e-6)
        train_conditions = torch.as_tensor(
            (train_condition_raw - condition_mean) / condition_std,
            dtype=torch.float32,
            device=DEVICE,
        )
        val_conditions = torch.as_tensor(
            (val_condition_raw - condition_mean) / condition_std,
            dtype=torch.float32,
            device=DEVICE,
        )
        train_states = torch.as_tensor(
            (train_target_raw - target_mean) / target_std,
            dtype=torch.float32,
            device=DEVICE,
        )
        val_states = torch.as_tensor(
            (val_target_raw - target_mean) / target_std,
            dtype=torch.float32,
            device=DEVICE,
        )
        model_state_dim, condition_dim = 2, 3
        reference_for_metrics = reference_positions

    normalization_metadata = (
        {
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        }
        if args.objective == "joint"
        else {
            "condition_mean": torch.as_tensor(condition_mean, dtype=torch.float32),
            "condition_std": torch.as_tensor(condition_std, dtype=torch.float32),
            "target_mean": torch.as_tensor(target_mean, dtype=torch.float32),
            "target_std": torch.as_tensor(target_std, dtype=torch.float32),
        }
    )
    checkpoint_metadata = {
        "model": args.model,
        "objective": args.objective,
        "data": str(args.data),
        "epochs": total_epochs,
        "max_samples": args.max_samples,
        "potential_type": potential_type,
        "period": args.period,
        "state_dim": model_state_dim,
        "condition_dim": condition_dim,
        "noise_levels": args.n_levels,
        "steps_per_level": args.steps_per_level,
        "ode_steps": args.ode_steps,
        **normalization_metadata,
    }
    sampler_name = (
        f"{args.n_levels}NoiseLevels_{args.steps_per_level}StepsPerNoiseLevel"
        if args.model == "diffusion"
        else f"{args.ode_steps}ODESteps"
    )
    total_epochs = args.epochs + len(
        resume_checkpoint.get("training_history", []) if resume_checkpoint else []
    )
    run_name = (
        f"{args.data.stem}_{args.model}_{args.objective}_{total_epochs}epochs_"
        f"{args.max_samples}samples_{sampler_name}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir = Path(__file__).resolve().parents[1] / "predictions"
    checkpoint_path = (
        output_dir / f"{run_name}_checkpoint.pt" if args.save_checkpoint else None
    )

    validation = (val_states, val_conditions)
    if args.model == "diffusion":
        model = Diffusion3D(model_state_dim, condition_dim).to(DEVICE)
        model, loss_history, validation_history = train_diffusion(
            model,
            train_states,
            train_conditions,
            args.epochs,
            validation=validation,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            resume_checkpoint=resume_checkpoint,
        )
        sample = sample_diffusion
        model_version = f"xy_t_diffusion_{args.objective}_v1"
        sampling_config = f"noise_levels={args.n_levels};steps_per_level={args.steps_per_level}"
    else:
        model = Flow3D(model_state_dim, condition_dim).to(DEVICE)
        model, loss_history, validation_history = train_flow(
            model,
            train_states,
            train_conditions,
            args.epochs,
            validation=validation,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            resume_checkpoint=resume_checkpoint,
        )
        sample = sample_flow
        model_version = f"xy_t_flow_{args.objective}_v1"
        sampling_config = f"ode_steps={args.ode_steps}"

    if potential_type == "chain":
        model_version += "_fractional_x_v1"
        sampling_config += f";fractional_x_period={args.period:g}"

    if args.objective == "joint":
        generated_normalized = (
            sample(
                model,
                args.generated_samples,
                n_levels=args.n_levels,
                steps_per_level=args.steps_per_level,
            )
            if args.model == "diffusion"
            else sample(model, args.generated_samples, ode_steps=args.ode_steps)
        )
        generated = generated_normalized.cpu().numpy() * std.numpy() + mean.numpy()
        if potential_type == "chain":
            generated[:, 0] = np.mod(generated[:, 0], 1.0) * args.period
        generated_for_metrics = generated
        generated_time = generated[:, 2]
        reference_time = times
    else:
        start_indices = rng.choice(
            len(times) - args.trajectory_length + 1, size=args.trajectories, replace=True,
        )
        trajectory_times = np.stack([
            times[start:start + args.trajectory_length] for start in start_indices
        ])
        generated = np.empty((args.trajectories, args.trajectory_length, 2), dtype=np.float32)
        generated[:, 0] = positions[start_indices, :2]
        if potential_type == "chain":
            generated[:, 0, 0] %= args.period
        for step in range(args.trajectory_length - 1):
            condition_raw = np.column_stack((
                generated[:, step], trajectory_times[:, step],
            )).astype(np.float32)
            if potential_type == "chain":
                condition_raw[:, 0] = (
                    np.mod(condition_raw[:, 0], args.period) / args.period
                )
            condition = torch.as_tensor(
                (condition_raw - condition_mean) / condition_std,
                dtype=torch.float32,
                device=DEVICE,
            )
            next_normalized = (
                sample(
                    model,
                    args.trajectories,
                    condition=condition,
                    n_levels=args.n_levels,
                    steps_per_level=args.steps_per_level,
                )
                if args.model == "diffusion"
                else sample(
                    model, args.trajectories, condition=condition, ode_steps=args.ode_steps,
                )
            )
            next_position = next_normalized.cpu().numpy() * target_std + target_mean
            if potential_type == "chain":
                next_position[:, 0] = np.mod(next_position[:, 0], 1.0) * args.period
            generated[:, step + 1] = next_position
        generated_for_metrics = generated.reshape(-1, 2)
        generated_time = trajectory_times
        reference_time = times
        sampling_config += (
            f";trajectories={args.trajectories};trajectory_length={args.trajectory_length}"
        )

    marginal_distance, joint_distance = histogram_distances(
        reference_for_metrics, generated_for_metrics,
    )
    reference_rate = generated_rate = None
    if args.objective == "transitions":
        reference_rate = _crossing_rate(positions[:, :2], times, potential_type, args.period)
        generated_rate = _crossing_rate(generated, generated_time, potential_type, args.period)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_name}_samples.png"
    loss_output = output_dir / f"{run_name}_loss.png"
    data_output = output_dir / f"{run_name}_generated.npz"
    _save_comparison(
        reference if args.objective == "joint" else reference_positions,
        generated if args.objective == "joint" else generated,
        reference_time,
        generated_time,
        args.objective,
        output,
    )
    plot_training_loss(loss_history, loss_output, validation_history)
    np.savez_compressed(
        data_output,
        generated=generated,
        generated_time=generated_time,
        objective=args.objective,
        model=args.model,
    )
    log_path = log_prediction(
        output,
        model_version=model_version,
        einstein_diffusion=None,
        barrier_height=None,
        kramers_rate=None,
        sampling_config=sampling_config,
        histogram_marginal_hellinger=marginal_distance,
        histogram_joint_hellinger=joint_distance,
        reference_crossing_rate=reference_rate,
        generated_crossing_rate=generated_rate,
    )
    print(f"Device: {DEVICE}")
    print(f"Mean marginal histogram Hellinger distance: {marginal_distance:.5f}")
    print(f"Joint histogram Hellinger distance: {joint_distance:.5f}")
    if reference_rate is not None:
        print(f"Observed MD crossing rate: {reference_rate:.6g} per time unit")
        print(f"Generated crossing rate: {generated_rate:.6g} per time unit")
    print(f"Generated data saved as '{data_output.resolve()}'")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Training loss saved as '{loss_output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")
    if checkpoint_path is not None:
        print(f"Training checkpoint saved as '{checkpoint_path.resolve()}'")


if __name__ == "__main__":
    main()
