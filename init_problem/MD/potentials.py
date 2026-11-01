import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



def compute_potential(x, y, potential_type, well_amplitude, box_length=32.0):

    
    is_chain = potential_type == "chain"
    sigma_x, sigma_y = 0.5, 0.5
    k_confine = k_y_confinement = 0.2
    well_separation = 2.0

    if is_chain:
        cell_spacing = 4.0
        n_cells = 8
        cell_centers = cell_spacing * np.arange(n_cells)
        well_x_centers = (cell_centers[:, None] + [-well_separation / 2, well_separation / 2]).ravel()
        well_amplitudes = np.tile([well_amplitude, well_amplitude], n_cells)

        dx = (x[..., None] - well_x_centers + box_length / 2) % box_length - box_length / 2
        dy = y[..., None]
        wells = well_amplitudes * np.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
        return -np.sum(wells, axis=-1) + 0.5 * k_y_confinement * y ** 2

    double_well_centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    dx, dy = x[..., None] - double_well_centers[:, 0], y[..., None] - double_well_centers[:, 1]
    wells = well_amplitude * np.exp(-0.5 * (dx ** 2 + dy ** 2) / sigma_x ** 2)
    return -np.sum(wells, axis=-1) + 0.5 * k_confine * (x ** 2 + y ** 2)