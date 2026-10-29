import torch
import torch.nn as nn

from .data import STATE_DIM

'''if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("M1 GPU is ready!")
'''
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    #print("CUDA GPU is ready!")
else:
    DEVICE = torch.device("cpu")
    #print("Defaulting to CPU.")


model_VERSION = "xy_v1"
SIGMA_MIN = 0.02
SIGMA_MAX = 1.0


class DiffusionModel(nn.Module):
    """Predict the score of a noisy position vector."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, X_tau, sigma_tau):
        inputs = torch.cat([X_tau, sigma_tau[:, None]], dim=1)
        return self.net(inputs)



class FlowModel(nn.Module):
    """Predict flow velocity using model-internal time tau."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, X_tau, tau):
        inputs = torch.cat([X_tau, tau[:, None]], dim=1)
        return self.net(inputs)