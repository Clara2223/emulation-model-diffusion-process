import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import get_potential_settings, load_equilibrium_bundle
from .checkpoint import (
    checkpoint_option,
    checkpoint_settings,
    load_training_checkpoint,
    resolve_checkpoint_data,
)

from .model_2D import (
    DEVICE,
    model_VERSION,
    DiffusionModel,
    FlowModel,
)

from .predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    histogram_distances,
    log_prediction,
    plot_training_loss,
    plot_results,
    sample_flow,
    sample_score,
)

from .train import train_flow, train_score

#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_data_100000000steps_50000burn.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "3D_double_well_data_100000000steps_50000burn.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_periodic_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
    
### double well
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "2D_double_well_1e8steps_5e4burn_20261009_1429.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_1e8steps_5e4burn_20261021_2056.npz"


#### chain of wells
DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_chain_1e8steps_5e4burn_20261021_2133.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "2D_chain_1e8steps_5e4burn_20261019_2112.npz"


#"double_well_data_100000000steps_50000burn_20260918_115938.npz"

def main():
    #inputs 
    parser = argparse.ArgumentParser(description="Train a 2D position diffusion or flow model.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--period", type=float)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--model", choices=("diffusion", "flow"))
    parser.add_argument("--n-levels", type=int)
    parser.add_argument(
        "--steps-per-level", type=int,
        help="Score-based Langevin correction updates at each noise level (diffusion)",
    )
    parser.add_argument(
        "--ode-steps", type=int,
        help="Numerical integration steps over the flow model's transport interval",
    )
    parser.add_argument("--diffusion-frames", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save model and optimizer state after training",
    )
    parser.add_argument("--show", action="store_true")
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
    args.data = args.data or DEFAULT_DATA
    args.model = checkpoint_option(parser, args.model, settings, "model", "diffusion")
    args.period = checkpoint_option(parser, args.period, settings, "period", 32.0)
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
    args.diffusion_frames = checkpoint_option(
        parser, args.diffusion_frames, settings, "diffusion_frames", 500_000,
    )
    if min(args.epochs, args.n_levels, args.steps_per_level, args.ode_steps) < 1:
        parser.error("--epochs, --n-levels, --steps-per-level, and --ode-steps must be positive")
    if args.generated_samples < 1:
        parser.error("--generated-samples must be at least 1")
    if args.period <= 0:
        parser.error("--period must be positive")

    #data handling: load data train and compute mean/std for normalization
    train_data, true_samples, positions, times = load_equilibrium_bundle(
        args.data, args.max_samples, args.diffusion_frames)
    if train_data.ndim != 2 or train_data.shape[1] < 2:
        parser.error("The data archive must contain positions with x and y columns")
    train_data = train_data[:, :2]
    true_samples = true_samples[:, :2]
    positions = positions[:, :2]
    potential_type, barrier_amplitude, barrier_width, well_amplitude = (
        get_potential_settings(args.data)
    )
    if potential_type == "chain":
        train_data[:, 0] = torch.remainder(train_data[:, 0], args.period) / args.period
        true_samples[:, 0] %= args.period

    if len(train_data) < 2:
        parser.error("--max-samples must select at least 2 samples for training and validation")
    validation_size = max(1, int(len(train_data) * 0.1))
    permutation = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(7))

    training_data = train_data[permutation[validation_size:]]
    validation_data = train_data[permutation[:validation_size]]
    mean = training_data.mean(dim=0).to(DEVICE)
    std = training_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
    train_loader = DataLoader(TensorDataset(training_data), batch_size=1000, shuffle=True)
    validation_loader = DataLoader(
        TensorDataset(validation_data), batch_size=1000, shuffle=False )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sampler_name = (
        f"{args.ode_steps}ODESteps"
        if args.model == "flow"
        else f"{args.n_levels}NoiseLevels_{args.steps_per_level}StepsPerNoiseLevel"
    )
    total_epochs = args.epochs + len(
        resume_checkpoint.get("training_history", []) if resume_checkpoint else []
    )
    run_name = (
        f"{args.data.stem}_{args.model}_{total_epochs}epochs_"
        f"{len(train_data)}samples_{sampler_name}_{run_timestamp}"
    )
    prediction_dir = Path(__file__).resolve().parents[1] / "predictions"
    checkpoint_path = (
        prediction_dir / f"{run_name}_checkpoint.pt" if args.save_checkpoint else None
    )
    checkpoint_metadata = {
        "model": args.model,
        "data": str(args.data),
        "epochs": total_epochs,
        "max_samples": len(train_data),
        "potential_type": potential_type,
        "period": args.period,
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "noise_levels": args.n_levels,
        "steps_per_level": args.steps_per_level,
        "ode_steps": args.ode_steps,
        "diffusion_frames": args.diffusion_frames,
    }

    #choose model type and train, generate samples
    if args.model == "flow":
        model, loss_history, validation_loss_history = train_flow(
            FlowModel().to(DEVICE), train_loader, mean, std,
            epochs=args.epochs,
            validation_loader=validation_loader,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            resume_checkpoint=resume_checkpoint,
        )
        generated = sample_flow(model, mean, std, args.generated_samples, args.ode_steps)
        sampling_config = f"ode_steps={args.ode_steps}"
    else:
        model, loss_history, validation_loss_history = train_score(
            DiffusionModel().to(DEVICE), train_loader, mean, std,
            epochs=args.epochs,
            validation_loader=validation_loader,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            resume_checkpoint=resume_checkpoint,
        )
        generated = sample_score(
            model, mean, std, args.generated_samples, args.n_levels, args.steps_per_level,
        )
        sampling_config = (
            f"noise_levels={args.n_levels};steps_per_level={args.steps_per_level}"
        )
    model_version = f"{model_VERSION}_{args.model}"
    if potential_type == "chain":
        model_version += "_fractional_x_v1"
        sampling_config += f";fractional_x_period={args.period:g}"
        generated[:, 0] = (generated[:, 0] % 1.0) * args.period
    

    #results: save plot, 'stats' and log file
    einstein = calculate_einstein_diffusion(positions,times)
    kramers_rate, barrier_height = calculate_kramers_rate(
        potential_type=potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        period=args.period,
    )
    marginal_distance, joint_distance = histogram_distances(true_samples, generated)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / f"{run_name}_samples.png"
    loss_output = prediction_dir / f"{run_name}_loss.png"
    plot_results(
        true_samples,
        generated,
        output,
        model_type=args.model,
        potential_type=potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        well_amplitude=well_amplitude,
        show=args.show,
        period=args.period,
    )
    plot_training_loss(
        loss_history, loss_output, validation_losses=validation_loss_history, show=args.show,
    )
    log_path = log_prediction(
        output,
        model_version=model_version,
        einstein_diffusion=einstein,
        barrier_height=barrier_height,
        kramers_rate=kramers_rate,
        sampling_config=sampling_config,
        histogram_marginal_hellinger=marginal_distance,
        histogram_joint_hellinger=joint_distance,
    )
    print(f"Device: {DEVICE}")
    print(f"Einstein diffusion coefficient: {einstein:.5f}")
    print(f"Kramers barrier height: {barrier_height:.5f}")
    print(f"Kramers transition rate: {kramers_rate:.5f}")
    print(f"Mean marginal histogram Hellinger distance: {marginal_distance:.5f}")
    print(f"Joint histogram Hellinger distance: {joint_distance:.5f}")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Training loss curve saved as '{loss_output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")
    if checkpoint_path is not None:
        print(f"Training checkpoint saved as '{checkpoint_path.resolve()}'")


if __name__ == "__main__":
    main()
