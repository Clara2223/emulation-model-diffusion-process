import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_training_checkpoint
from .model_2D import DEVICE, SIGMA_MAX, SIGMA_MIN


class Diffusion3D(nn.Module):
    """Noise-conditioned score model for 3D samples or conditional transitions."""

    def __init__(
        self,
        state_dim=3,
        condition_dim=0,
        hidden_dim=64,
        embedding_dim=16,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        frequencies = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), embedding_dim // 2))
        self.register_buffer("frequencies", frequencies)
        self.net = nn.Sequential(
            nn.Linear(
                state_dim + condition_dim + embedding_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, X_tau, sigma_tau, condition=None):
        log_sigma = torch.log(sigma_tau.clamp_min(1e-6)).unsqueeze(1)
        angles = log_sigma * self.frequencies.unsqueeze(0)
        features = [X_tau]
        if self.condition_dim:
            if condition is None or condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected a condition with {self.condition_dim} features")
            features.append(condition)
        features.append(torch.cat((torch.sin(angles), torch.cos(angles)), dim=1))
        return self.net(torch.cat(features, dim=1))


class Flow3D(nn.Module):
    """Time-conditioned flow model for 3D samples or conditional transitions."""

    def __init__(
        self,
        state_dim=3,
        condition_dim=0,
        hidden_dim=64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(
                state_dim + condition_dim + 1,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, X_tau, tau, condition=None):
        features = [X_tau, tau[:, None]]
        if self.condition_dim:
            if condition is None or condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected a condition with {self.condition_dim} features")
            features.append(condition)
        return self.net(torch.cat(features, dim=1))


def _fit(
    model,
    states,
    conditions,
    epochs,
    batch_size,
    loss_fn,
    validation,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    if len(states) == 0:
        raise ValueError("Cannot train on an empty dataset")
    if conditions is not None and len(conditions) != len(states):
        raise ValueError("Condition and target counts must match")
    tensors = (states,) if conditions is None else (states, conditions)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    validation_loader = None
    if validation is not None:
        validation_states, validation_conditions = validation
        if len(validation_states) == 0:
            raise ValueError("Cannot validate with an empty data set")
        validation_tensors = (
            (validation_states,) if validation_conditions is None
            else (validation_states, validation_conditions)
        )
        validation_loader = DataLoader(
            TensorDataset(*validation_tensors), batch_size=batch_size,
        )

    def batch_loss(batch):
        Z = batch[0].to(DEVICE)
        condition = batch[1].to(DEVICE) if conditions is not None else None
        return loss_fn(Z, condition)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    validation_history = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history = list(resume_checkpoint.get("training_history", []))
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
            loss = batch_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = len(batch[0])
            total_loss += loss.detach().item() * batch_size
            sample_count += batch_size
        history.append(total_loss / sample_count)

        if validation_loader is not None:
            model.eval()
            validation_total_loss = 0.0
            validation_sample_count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    loss = batch_loss(batch)
                    batch_size = len(batch[0])
                    validation_total_loss += loss.item() * batch_size
                    validation_sample_count += batch_size
            validation_history.append(
                validation_total_loss / validation_sample_count,
            )

    if checkpoint_path is not None:
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            checkpoint_metadata or {},
            history,
            validation_history,
        )

    model.train()
    return model, history, validation_history


def train_diffusion(
    model,
    states,
    conditions=None,
    epochs=100,
    batch_size=1000,
    validation=None,
    *,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    def loss_fn(Z, condition):
        tau = 1e-6 + (1 - 1e-6) * torch.rand(len(Z), device=DEVICE)  # keep tau above zero
        sigma_tau = SIGMA_MIN * torch.exp(tau * np.log(SIGMA_MAX / SIGMA_MIN))
        E = torch.randn_like(Z)
        X_tau = Z + sigma_tau[:, None] * E
        score = model(X_tau, sigma_tau, condition)
        target_score = -E / sigma_tau[:, None]
        return (((score - target_score) * sigma_tau[:, None]) ** 2).sum(dim=1).mean()

    return _fit(
        model, states, conditions, epochs, batch_size, loss_fn, validation,
        checkpoint_path, checkpoint_metadata, resume_checkpoint,
    )


def train_flow(
    model,
    states,
    conditions=None,
    epochs=100,
    batch_size=1000,
    validation=None,
    *,
    checkpoint_path=None,
    checkpoint_metadata=None,
    resume_checkpoint=None,
):
    def loss_fn(Z, condition):
        E = torch.randn_like(Z)
        tau = torch.rand(len(Z), device=DEVICE)
        X_tau = (1 - tau[:, None]) * E + tau[:, None] * Z
        target_velocity = Z - E
        velocity = model(X_tau, tau, condition)
        return ((velocity - target_velocity) ** 2).sum(dim=1).mean()

    return _fit(
        model, states, conditions, epochs, batch_size, loss_fn, validation,
        checkpoint_path, checkpoint_metadata, resume_checkpoint,
    )


def sample_diffusion(model, n_samples, condition=None, n_levels=12, steps_per_level=50):
    model.eval()
    X = torch.randn(n_samples, model.state_dim, device=DEVICE)
    sigmas = torch.exp(torch.linspace(np.log(1.0), np.log(0.02), n_levels, device=DEVICE))
    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = sigma.expand(n_samples)
            step_size = 0.02 * sigma**2
            for _ in range(steps_per_level):
                # Euler-Maruyama Langevin step: score drift plus Gaussian noise.
                X = X + step_size * model(X, sigma_batch, condition)
                X = X + torch.sqrt(2 * step_size) * torch.randn_like(X)
    return X


def sample_flow(model, n_samples, condition=None, ode_steps=50):
    model.eval()
    X = torch.randn(n_samples, model.state_dim, device=DEVICE)
    step_size = 1.0 / ode_steps
    with torch.no_grad():
        for step in range(ode_steps):
            tau = torch.full((n_samples,), step / ode_steps, device=DEVICE)
            X = X + step_size * model(X, tau, condition)
    return X
