from math import log

import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_training_checkpoint
from .model_2D import DEVICE, SIGMA_MAX, SIGMA_MIN


def _fit(
    model,
    loader,
    validation_loader,
    training_loss,
    validation_loss,
    epochs,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    if len(loader.dataset) == 0:
        raise ValueError("Cannot train with an empty data loader")
    if validation_loader is not None and len(validation_loader.dataset) == 0:
        raise ValueError("Cannot validate with an empty data loader")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_history = []
    validation_history = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        training_history = list(resume_checkpoint.get("training_history", []))
        validation_history = list(resume_checkpoint.get("validation_history", []))
        if "torch_rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and resume_checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_states"])
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in loader:
            loss = training_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = len(batch[0])
            total_loss += loss.detach().item() * batch_size
            sample_count += batch_size
        if sample_count == 0:
            raise ValueError("Cannot train with an empty data loader")
        training_history.append(total_loss / sample_count)

        if validation_loader is not None:
            model.eval()
            validation_total_loss = 0.0
            validation_sample_count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    loss = validation_loss(batch)
                    batch_size = len(batch[0])
                    validation_total_loss += loss.item() * batch_size
                    validation_sample_count += batch_size
            if validation_sample_count == 0:
                raise ValueError("Cannot validate with an empty data loader")
            validation_history.append(
                validation_total_loss / validation_sample_count,
            )

    if checkpoint_path is not None:
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            checkpoint_metadata or {},
            training_history,
            validation_history,
        )

    model.train()
    return model, training_history, validation_history


def train_score(
    model,
    loader,
    mean,
    std,
    *,
    epochs,
    validation_loader,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    # Train the score model on X_tau = Z + sigma_tau * E.
    def score_loss(Z, sigma_tau, E):
        X_tau = Z + sigma_tau[:, None] * E
        score = model(X_tau, sigma_tau)
        target = -E / sigma_tau[:, None]
        return (((score - target) * sigma_tau[:, None]) ** 2).sum(dim=1).mean()

    def training_loss(batch):
        (Z,) = batch         #versaler er batches, små er single data samples
        Z = (Z.to(DEVICE) - mean) / std
        tau = 1e-6 + (1 - 1e-6) * torch.rand(len(Z), device=DEVICE)  # tau above zero for numerical stability
        sigma_tau = SIGMA_MIN * torch.exp(tau * log(SIGMA_MAX / SIGMA_MIN))    #'exploding variance method'
        E = torch.randn_like(Z)
        return score_loss(Z, sigma_tau, E)

    Z_validation = torch.cat([batch[0] for batch in validation_loader]).to(DEVICE)
    Z_validation = (Z_validation - mean) / std
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    tau_validation = 1e-6 + (1 - 1e-6) * torch.rand(
        len(Z_validation), device=DEVICE, generator=generator,
    )  # keep tau above zero
    sigma_tau_validation = SIGMA_MIN * torch.exp(
        tau_validation * log(SIGMA_MAX / SIGMA_MIN),
    )
    E_validation = torch.randn(Z_validation.shape, device=DEVICE, generator=generator)
    validation_loader = DataLoader(
        TensorDataset(Z_validation, sigma_tau_validation, E_validation),
        batch_size=validation_loader.batch_size or len(Z_validation),
    )
    return _fit(
        model, loader, validation_loader, training_loss,
        lambda batch: score_loss(*batch), epochs, checkpoint_path,
        checkpoint_metadata, resume_checkpoint,
    )


def train_flow(
    model,
    loader,
    mean,
    std,
    *,
    epochs,
    validation_loader,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    """Train the flow model on random noise-to-data interpolation points."""
    def flow_loss(Z, E, tau):
        X_tau = (1 - tau[:, None]) * E + tau[:, None] * Z
        target_velocity = Z - E
        velocity = model(X_tau, tau)
        return ((velocity - target_velocity) ** 2).sum(dim=1).mean()

    def training_loss(batch):
        (Z,) = batch
        Z = (Z.to(DEVICE) - mean) / std
        E = torch.randn_like(Z)
        tau = torch.rand(len(Z), device=DEVICE)
        return flow_loss(Z, E, tau)

    Z_validation = torch.cat([batch[0] for batch in validation_loader]).to(DEVICE)
    Z_validation = (Z_validation - mean) / std
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    E_validation = torch.randn(Z_validation.shape, device=DEVICE, generator=generator)
    tau_validation = torch.rand(len(Z_validation), device=DEVICE, generator=generator)
    validation_loader = DataLoader(
        TensorDataset(Z_validation, E_validation, tau_validation),
        batch_size=validation_loader.batch_size or len(Z_validation),
    )
    return _fit(
        model, loader, validation_loader, training_loss,
        lambda batch: flow_loss(*batch), epochs, checkpoint_path,
        checkpoint_metadata, resume_checkpoint,
    )
