from pathlib import Path
import numpy as np
import torch

STATE_DIM = 2

def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    """Select a chronological, evenly distributed subset before any shuffling."""
    if total < 0 or count < 0:
        raise ValueError("Sample counts cannot be negative")
    if count > total:
        raise ValueError("Cannot select more indices than the available samples")
    if count == total:
        return np.arange(total)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return (np.arange(count, dtype=np.int64) * total // count)


def _position_trajectories(positions: np.ndarray) -> np.ndarray:
    if positions.ndim == 2:
        return positions[None, ...]
    if positions.ndim == 3:
        return positions
    raise ValueError("Position data must have shape (frames, dim) or (trajectories, frames, dim)")


def get_potential_settings(path: Path) -> tuple[str, float, float, float]:
    """Read potential metadata, detecting legacy archive types from their names."""
    with np.load(path, allow_pickle=True) as data:
        if "potential_type" in data:
            name = str(data["potential_type"].item()).lower()
        else:
            name = Path(path).stem.lower()
        barrier_amplitude = (
            float(data["barrier_amplitude"].item()) if "barrier_amplitude" in data else 0.0
        )
        barrier_width = (
            float(data["barrier_width"].item()) if "barrier_width" in data else 0.25
        )
        '''well_amplitude = (
            data["well_amplitude"].item() if "well_amplitude" in data else None
            float(val) if val is not None else 3.0
            #float(data["well_amplitude"].item()) if "well_amplitude" in data else 3.0
        )'''
        well_amplitude = float(data["well_amplitude"].item()) if ("well_amplitude" in data and data["well_amplitude"].item() is not None) else 3.0

    if "chain" in name:
        return "chain", barrier_amplitude, barrier_width, well_amplitude
    if "hbarrier" in name or "h_barrier" in name:
        if barrier_amplitude <= 0:
            raise ValueError(
                f"Archive identifies an H-barrier but has no barrier parameters: {path}"
            )
        return "double-well-Hbarrier", barrier_amplitude, barrier_width, well_amplitude
    if "double-well" in name or "double_well" in name:
        return "double-well", barrier_amplitude, barrier_width, well_amplitude
    raise ValueError(f"Cannot determine potential type for data archive: {path}")


def get_potential_type(path: Path) -> str:
    """Compatibility helper returning only the potential type."""
    return get_potential_settings(path)[0]


def load_equilibrium_data(path: Path, max_samples: int, seed: int = 7):
    """Load evenly spaced two-dimensional position samples."""
    with np.load(path) as data:
        positions = _position_trajectories(data["positions"])
        positions = positions.reshape(-1, positions.shape[-1])
        count = min(max_samples, len(positions))
        indices = evenly_spaced_indices(len(positions), count)
        samples = positions[indices]
    return torch.from_numpy(samples.astype(np.float32, copy=False))


def load_equilibrium_bundle(
    path: Path,
    max_samples: int,
    diffusion_frames: int,
    seed: int = 7,
    trajectory: int = 1,
):
    """Read samples and references from one selected trajectory."""
    with np.load(path) as data:
        all_positions = _position_trajectories(data["positions"])
        if trajectory < 1 or trajectory > len(all_positions):
            raise ValueError(f"trajectory must be between 1 and {len(all_positions)}")
        # keep the model on one initial position because it has no trajectory id
        positions = all_positions[trajectory - 1:trajectory]
        times = data["md_time"]
        if times.ndim != 1 or len(times) != positions.shape[1]:
            raise ValueError("The md_time array must match the saved frame count")
        samples_all = positions.reshape(-1, positions.shape[-1])
        count = min(max_samples, len(samples_all))
        indices = evenly_spaced_indices(len(samples_all), count)
        samples = samples_all[indices]
        true_samples = samples_all.copy()
        diffusion_positions = positions[0, :diffusion_frames].copy()
        diffusion_times=times[:diffusion_frames].copy()
    return (
        torch.from_numpy(samples.astype(np.float32, copy=False)),   #train data
        true_samples,
        diffusion_positions,
        diffusion_times,
    )
    
    


def load_training_bundle(path: Path, max_samples: int, diffusion_frames: int, seed: int = 7):
    """Read the archive once and prepare training and diagnostic arrays."""
    with np.load(path) as data:
        positions = data["positions"]
        velocities = data["velocities"]
        energies = data["energy"][:, None]
        rng = np.random.default_rng(seed)
        count = min(max_samples, len(positions) - 1)
        indices = rng.choice(len(positions) - 1, size=count, replace=False)
        current = np.concatenate((positions[indices], velocities[indices], energies[indices]), axis=1)
        following = np.concatenate(
            (positions[indices + 1], velocities[indices + 1], energies[indices + 1]), axis=1
        )
        train_data = torch.from_numpy(current.copy())
        true_samples = np.column_stack(
            (positions[:5000], velocities[:5000], energies[:5000, 0])
        )
        diffusion_positions = positions[:diffusion_frames].copy()
        diffusion_velocities = velocities[:diffusion_frames].copy()

    return (
        train_data,
        torch.from_numpy(current.copy()),
        torch.from_numpy(following.copy()),
        true_samples,
        diffusion_positions,
        diffusion_velocities,
    )
