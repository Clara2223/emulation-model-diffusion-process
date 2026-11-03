import argparse
from datetime import datetime
from pathlib import Path
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
from .predict import (
    histogram_distances,
    log_prediction,
    plot_training_loss,
    plot_trajectory_comparison,
)


#paths
### double well
    #DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_1e8steps_5e4burn_20261021_2056.npz"
#DEFAULT_DATA =Path(__file__).resolve().parents[1] /"MD"/ "train_data" /'3D_double_well_1e8steps_0e0burn_20261101_2142.npz'

#### chain of wells
    #DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_chain_1e8steps_5e4burn_20261021_2133.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / '3D_chain_1e8steps_0e0burn_20261102_1739.npz'
DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_adjusted_1e8steps_0e0burn_20261103_1649_4trans.npz"


### high barrier double well
    #DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_adjusted_1e8steps_0e0burn_20261030_1446.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / '3D_double_well_adjusted_1e8steps_0e0burn_20261102_0733.npz'



def main():
    parser = argparse.ArgumentParser(
        description="Train a joint, condition-based, or AdaLN 3D trajectory model",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--period", type=float)
    parser.add_argument("--objective", choices=("joint", "condition", "AdaLN"))
    parser.add_argument("--model", choices=("diffusion", "flow"))
    parser.add_argument(
        "--trajectory", type=int, nargs="+",
        help="1-based trajectories to train on (default: all for conditional objectives, 1 for joint)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--n-levels", type=int)
    parser.add_argument( "--steps-per-level", type=int, help="Score-based Langevin updates at each noise level (diffusion)")
    parser.add_argument( "--ode-steps", type=int,help="Numerical integration steps over the flow transport interval")
    parser.add_argument("--save-checkpoint",action="store_true",help="Save model and optimizer state after training")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    resume_checkpoint = None
    settings = {}
    
    # DATA 
    if args.resume_checkpoint is not None:
        resume_checkpoint = load_training_checkpoint(args.resume_checkpoint)
        settings = checkpoint_settings(args.resume_checkpoint, resume_checkpoint)
        if args.data is None:
            args.data = resolve_checkpoint_data(
                args.resume_checkpoint, settings, Path(__file__).resolve().parents[1],
            )
    if args.data is None:
        args.data = DEFAULT_DATA
        print(f"--data is required unless --resume-checkpoint is used. Using default data: {DEFAULT_DATA}")
    args.period = checkpoint_option(parser, args.period, settings, "period", 32.0)
    
    # MODEL AND TRAINING OPTIONS
    args.objective = checkpoint_option(
        parser, args.objective, settings, "objective", "joint",
    )
    if args.objective not in {"joint", "condition", "AdaLN"}:
        parser.error(f"Unsupported checkpoint objective: {args.objective}")
    saved_trajectories = settings.get("trajectory")
    if args.trajectory is None:
        if saved_trajectories is not None:
            args.trajectory = (
                [saved_trajectories]
                if isinstance(saved_trajectories, int)
                else list(saved_trajectories)
            )
        else:
            args.trajectory = None if args.objective in {"condition", "AdaLN"} else [1]
    elif saved_trajectories is not None:
        saved_trajectories = (
            [saved_trajectories]
            if isinstance(saved_trajectories, int)
            else list(saved_trajectories)
        )
        if args.trajectory != saved_trajectories:
            parser.error(
                f"--trajectory must match the resumed checkpoint ({saved_trajectories})"
            )
    args.model = checkpoint_option(parser, args.model, settings, "model", "diffusion")
    args.max_samples = checkpoint_option(
        parser, args.max_samples, settings, "max_samples", 100_000, "samples",
    )
    
    # SAMPLING PARAMS 
    args.n_levels = checkpoint_option(
        parser, args.n_levels, settings, "n_levels", 20, "noise_levels",
    )
    args.steps_per_level = checkpoint_option(
        parser, args.steps_per_level, settings, "steps_per_level", 100,
    )
    args.ode_steps = checkpoint_option(parser, args.ode_steps, settings, "ode_steps", 50)
    
    # CHECKS 
    if min(args.epochs, args.max_samples, args.generated_samples,
           args.n_levels, args.steps_per_level, args.ode_steps) < 1:
        parser.error("training, sampling, and data-count arguments must be positive")
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2 for train/validation splits")
    if args.period <= 0:
        parser.error("--period must be positive")
    if args.trajectory is not None and (
        any(trajectory < 1 for trajectory in args.trajectory)
        or len(set(args.trajectory)) != len(args.trajectory)
    ):
        parser.error("--trajectory values must be unique positive integers")
    
    with np.load(args.data) as archive:
        positions = np.asarray(archive["positions"], dtype=np.float32)
        times = np.asarray(archive["md_time"], dtype=np.float32)
        initial_positions = (
            np.asarray(archive["initial_positions"], dtype=np.float32)
            if "initial_positions" in archive else None)
    if positions.ndim == 2:
        positions = positions[None, ...]     #broadcast to (1, n_saved, 2)
    n_trajectories, n_saved, _ = positions.shape
    '''
    # i create the data so i dont need to check this
    if positions.ndim != 3 or positions.shape[2] < 2:
        parser.error("The archive must contain 2D or 3D position samples with x and y columns")
    '''
    if times.ndim != 1 or len(times) != n_saved:
        parser.error("The archive's md_time array must match the saved frame count")
    if n_saved < 3 or not np.isfinite(positions[..., :2]).all() or not np.isfinite(times).all():
        parser.error("The archive needs at least three finite position/time samples")
    
    if args.objective in {"condition", "AdaLN"}:
        if initial_positions is None:
            parser.error("The condition and AdaLN objectives require initial_positions; regenerate the archive with MD/MD_sim.py")
        if (initial_positions.shape != (n_trajectories, 2)
            or not np.isfinite(initial_positions).all()
        ):
            parser.error("initial_positions must contain one finite x,y pair per trajectory")
    if args.trajectory is None:
        args.trajectory = list(range(1, n_trajectories + 1))
    if any(trajectory > n_trajectories for trajectory in args.trajectory):
        parser.error(f"--trajectory values must be between 1 and {n_trajectories}")
    if args.objective == "joint" and len(args.trajectory) != 1:
        parser.error("The joint objective supports one trajectory; select one with --trajectory")
    trajectory_indices = np.asarray(args.trajectory, dtype=np.int64) - 1
    positions = positions[trajectory_indices]
    if initial_positions is not None:
        initial_positions = initial_positions[trajectory_indices]
    n_trajectories = len(trajectory_indices)

    (
        potential_type,
        barrier_amplitude,
        barrier_width,
        well_amplitude,
    ) = get_potential_settings(args.data)
    reference_positions = positions[..., :2].copy()
    if potential_type == "chain":
        reference_positions[..., 0] %= args.period
    rng = np.random.default_rng(7)

    if args.objective == "joint":
        # use one trajectory because joint samples have no trajectory id
        reference = np.column_stack((reference_positions[0], times)).astype(np.float32)
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
        # Emulation objective: conditions = [time, initial_position_x, initial_position_y]
        times_flat = np.broadcast_to(times[None, :], (n_trajectories, n_saved)).reshape(-1)
        init_pos_flat = np.broadcast_to(initial_positions[:, None, :], (n_trajectories, n_saved, 2)).reshape(-1, 2)
        condition_raw = np.column_stack((times_flat, init_pos_flat)).astype(np.float32)
        target_raw = reference_positions.reshape(-1, 2).astype(np.float32)

        if potential_type == "chain":
            condition_raw[:, 1] = np.mod(condition_raw[:, 1], args.period) / args.period
            target_raw[:, 0] = np.mod(target_raw[:, 0], args.period) / args.period

        pairs = min(args.max_samples, len(target_raw))
        selected = evenly_spaced_indices(len(target_raw), pairs)
        
        train_condition_raw_all = condition_raw[selected]
        train_target_raw_all = target_raw[selected]

        # Random shuffling split
        permutation = torch.randperm(pairs, generator=torch.Generator().manual_seed(7))
        validation_count = max(1, int(pairs * 0.1))
        train_indices = permutation[validation_count:].numpy()
        val_indices = permutation[:validation_count].numpy()

        train_condition_raw = train_condition_raw_all[train_indices]
        val_condition_raw = train_condition_raw_all[val_indices]
        train_target_raw = train_target_raw_all[train_indices]
        val_target_raw = train_target_raw_all[val_indices]

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
        reference_for_metrics = reference_positions.reshape(-1, 2)

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
    
    
    
    # LOGS
    total_epochs = args.epochs + len(
        resume_checkpoint.get("training_history", []) if resume_checkpoint else []
    )
    checkpoint_metadata = {
        "model": args.model,
        "objective": args.objective,
        "data": str(args.data),
        "trajectory": args.trajectory,
        "epochs": total_epochs,
        "max_samples": args.max_samples,
        "generated_samples": args.generated_samples,
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
    run_name = (
        f"{args.data.stem}_{args.model}_{args.objective}_traj{'-'.join(map(str, args.trajectory))}_{total_epochs}epochs_"
        f"{args.max_samples}samples_{sampler_name}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir = Path(__file__).resolve().parents[1] / "predictions"
    checkpoint_path = (
        output_dir / f"{run_name}_checkpoint.pt" if args.save_checkpoint else None
    )


    # TRAINING actually 
    validation = (val_states, val_conditions)
    if args.model == "diffusion":
        model = Diffusion3D(
            model_state_dim, condition_dim, adaln=args.objective == "AdaLN",
        ).to(DEVICE)
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
        model = Flow3D(
            model_state_dim, condition_dim, adaln=args.objective == "AdaLN",
        ).to(DEVICE)
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
        sample_indices = rng.choice(
            len(train_condition_raw_all), size=args.generated_samples, replace=True,
        )
        generated_conditions = train_condition_raw_all[sample_indices]
        condition = torch.as_tensor(
            (generated_conditions - condition_mean) / condition_std,
            dtype=torch.float32,
            device=DEVICE,
        )
        generated_normalized = (
            sample(
                model,
                args.generated_samples,
                condition=condition,
                n_levels=args.n_levels,
                steps_per_level=args.steps_per_level,
            )
            if args.model == "diffusion"
            else sample(
                model, args.generated_samples, condition=condition, ode_steps=args.ode_steps,
            )
        )
        generated = generated_normalized.cpu().numpy() * target_std + target_mean
        if potential_type == "chain":
            generated[:, 0] = np.mod(generated[:, 0], 1.0) * args.period
        generated_for_metrics = generated
        generated_time = generated_conditions[:, 0]
        reference_time = times_flat
        sampling_config += f";emulation_samples={args.generated_samples}"

    marginal_distance, joint_distance = histogram_distances(
        reference_for_metrics, generated_for_metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_name}_samples.png"
    loss_output = output_dir / f"{run_name}_loss.png"
    data_output = output_dir / f"{run_name}_generated.npz"
    plot_trajectory_comparison(
        reference if args.objective == "joint" else reference_positions.reshape(-1, 2),
        generated,
        reference_time,
        generated_time,
        args.objective,
        output,
        trajectory_positions=reference_positions,
        trajectory_times=times,
        trajectory_count=len(positions),
        model=model,
        model_type=args.model,
        field_config={
            "state_dim": model_state_dim,
            "condition_dim": condition_dim,
            "state_mean": target_mean if args.objective != "joint" else mean.cpu().numpy(),
            "state_std": target_std if args.objective != "joint" else std.cpu().numpy(),
            **(
                {
                    "condition_mean": condition_mean,
                    "condition_std": condition_std,
                }
                if args.objective != "joint"
                else {}
            ),
        },
        potential_type=potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        period=args.period,
        well_amplitude=well_amplitude,
        training_positions=reference_positions,
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
    )
    print(f"Device: {DEVICE}")
    print(f"Mean marginal histogram Hellinger distance: {marginal_distance:.5f}")
    print(f"Joint histogram Hellinger distance: {joint_distance:.5f}")
    print(f"Generated data saved as '{data_output.resolve()}'")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Training loss saved as '{loss_output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")
    if checkpoint_path is not None:
        print(f"Training checkpoint saved as '{checkpoint_path.resolve()}'")


if __name__ == "__main__":
    main()
