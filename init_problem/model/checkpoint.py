import re
from pathlib import Path

import torch


_CHECKPOINT_NAME = re.compile(
    r"(?P<data>.+)_(?P<model>diffusion|flow)_"
    r"(?:(?P<objective>joint|transitions)_)?"
    r"(?P<epochs>\d+)epochs_(?P<samples>\d+)samples_"
    r"(?:(?P<noise_levels>\d+)NoiseLevels_"
    r"(?P<steps_per_level>\d+)StepsPerNoiseLevel|"
    r"(?P<ode_steps>\d+)ODESteps)_\d{8}_\d{6}_checkpoint"
)


def load_training_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {path}")
    return checkpoint


def checkpoint_settings(path: Path, checkpoint: dict) -> dict:
    match = _CHECKPOINT_NAME.fullmatch(path.stem)
    settings = match.groupdict() if match else {}
    settings = {key: value for key, value in settings.items() if value is not None}
    for key in ("epochs", "samples", "noise_levels", "steps_per_level", "ode_steps"):
        if key in settings:
            settings[key] = int(settings[key])
    return {**settings, **checkpoint.get("metadata", {})}


def checkpoint_option(parser, requested, settings, key, default, filename_key=None):
    saved = settings.get(key)
    if saved is None and filename_key:
        saved = settings.get(filename_key)
    if requested is not None and saved is not None and requested != saved:
        option_name = filename_key if key == "n_levels" else key
        parser.error(
            f"--{option_name.replace('_', '-')} must match the resumed checkpoint ({saved})"
        )
    return requested if requested is not None else saved if saved is not None else default


def resolve_checkpoint_data(path: Path, settings: dict, project_dir: Path) -> Path:
    saved_path = settings.get("data")
    candidates = []
    if saved_path:
        saved_path = Path(saved_path)
        candidates.extend((saved_path, project_dir / saved_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    stem = Path(settings["data"]).stem if settings.get("data") else None
    if stem and Path(stem).suffix:
        stem = Path(stem).stem
    if stem is None:
        match = _CHECKPOINT_NAME.fullmatch(path.stem)
        stem = match.group("data") if match else None
    if stem is not None:
        matches = sorted((project_dir / "MD" / "train_data").glob(f"{stem}.npz"))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"Cannot locate the training data for {path.name}; pass its path with --data"
    )


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
    training_history: list[float],
    validation_history: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
            "training_history": training_history,
            "validation_history": validation_history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        path,
    )
