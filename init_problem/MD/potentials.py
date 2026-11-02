import numpy as np


CHAIN_PERIOD = 32.0


def compute_potential(
    x,
    y,
    potential_type="double-well",
    barrier_amplitude=0.0,
    barrier_width=0.25,
    period=CHAIN_PERIOD,
    well_amplitude=3.0,
):
    if potential_type == "chain":
        centers = (period / 8 * np.arange(8)[:, None] + [-1.0, 1.0]).ravel()
        dx = (np.asarray(x)[..., None] - centers + period / 2) % period - period / 2
        dy = np.asarray(y)[..., None]
        wells = well_amplitude * np.exp(-0.5 * ((dx / 0.5) ** 2 + (dy / 0.5) ** 2))
        return -np.sum(wells, axis=-1) + 0.1 * np.asarray(y) ** 2

    if potential_type not in {"double-well", "double-well-Hbarrier"}:
        raise ValueError(f"Unsupported potential type: {potential_type}")
    g1 = well_amplitude * np.exp(-((x + 1.0) ** 2 + y**2) / (2 * 0.5**2))
    g2 = well_amplitude * np.exp(-((x - 1.0) ** 2 + y**2) / (2 * 0.5**2))
    ridge = 0.0
    if potential_type == "double-well-Hbarrier":
        if barrier_amplitude <= 0 or barrier_width <= 0:
            raise ValueError("H-barrier requires positive amplitude and width metadata")
        ridge = barrier_amplitude * np.exp(-0.5 * (np.asarray(x) / barrier_width) ** 2)
    return -(g1 + g2) + 0.1 * (x**2 + y**2) + ridge
