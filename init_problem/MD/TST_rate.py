"""Transition-rate estimates in the reduced units used by MD_sim.py.

Examples:
  python MD/TST_rate.py --target-transitions 1 --n-steps 100000000 --burn-in 50000
  python MD/TST_rate.py --barrier-height 5 --production-time 1000
  python MD/TST_rate.py --method pmf --data MD/train_data/example.npz
"""

import argparse
import numpy as np


# Calculating the rate constant of a chemical reaction using Transition State Theory (TST)
##---------------------------------------------------------------------
##---------------------------------------------------------------------
##1 : Harmonic approximation of the potential energy surface (PES)
# this is only a harmonic approx. since the well is gaussian i.e. not quadratic.
#I calculate the gibbs free energy with thermodynamic relation
#
#hessian matrix construction
# - the second derivative of the potential energy V wrt. the coordinates x & y;
# Hxy=delta**2 V/delta x delta y
#
#freq. determination
# - the eigenvalues of the hessian matrix are the vibrational frequencies of the system
# omega_x, omega_y = sqrt(eigenvalues(Hxy)/m)
# nu_x, nu_y = omega_x/(2 pi), omega_y/(2 pi)
#
#vibrational partition function
# q_vib= product_i [ exp(-h nu_i / 2 k_B T) /(1-exp(-h nu_i/kT)) ]
#then these vibrations are used to express the entropy with the relation S=k_B ln(q_vib)+k_B T (delta ln(q_vib)/delta T)
# S=k_B ln(q_vib)+k_B T (delta ln(q_vib)/delta T)
# the enthalpy can be expressed /approximated by the potential energy at the minimum of the well, i.e. H=V_min difference to the transition state, which is the barrier height equivalent energy in the PES.
# + the the zero point correction ?
# H=V_min - V_TS + ?1/2 h (nu_x + nu_y)?
# T I defined for the MD sim.
# G=delta_H - T delta_S
#
#Lastly the Eyrin TST rate constant:
# k_TST = (k_B T / h) exp(-delta_G / k_B T)
##---------------------------------------------------------------------
#2 : Probability density & MD sampling
##---------------------------------------------------------------------
# using my MD sampling, one obtaisn the more 'correct' free energy surface, which is not harmonic.
#
#potential of mean force (PMF) / Free energy profile
# P(x) probability density of the system along the reaction coordinate x (which is obtained from the MD sampling) to free energy G(x) with the relation:
# G(x)=-k_BT ln(P(x)) + C
#
#free energy difference: comparing prob. at TS point against the integrated prob. density over the reactant well
# Delta_G=-k_BT*ln(P(x_TS)/integral_well(P(x)dx))
#
#rate constant from prob. int: rate const. prop. to ratio of the prob. at TS to the integrated prob. density over the reactant well - often coupled w. avg. velocity crossing terms in dynamical corrections
# k=v_x_abs_avg * P(x_TS)/integral_well(P(x)dx)
#
##---------------------------------------------------------------------
#3 : very rough approx: Using just the energy difference TS-V_min as E_act in Arrhenius
##---------------------------------------------------------------------
# this would be the approximation that the entropy does not matter / is negligable
# delta_E=V_min-V_TST
# k_tst= (k_B T / h) exp(-delta_E / k_B T)
#
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#then for a poisson process (like a rare-event transition), the expected waiting time for the first transition is the inverse of the rate:
# T_TS=1/k
#
# expected_transitions== N = k * T
#
# total production time of MD simulation= (n_steps - burn_in) * dt
# safe_production_time = 2 * (n_steps - burn_in) * dt
#        meaning it calculates the double to be sure we have the requested one rate state transition
# because: MD is random - statistical fluctuations may cause less transitions that hoping for otherwise.
#
# Quantum Eyring expressions are notes only: MD_sim.py uses reduced classical units,
# with no physical conversion for Planck's constant. Rates here use classical TST.


def _terms(x, amplitude, well_width, confinement):
    offsets = np.array([x + 1, x - 1])
    wells = np.exp(-offsets**2 / (2 * well_width**2))
    energy = -amplitude * wells.sum() + confinement * x**2 / 2
    gradient = amplitude * np.sum(offsets * wells) / well_width**2 + confinement * x
    curvature_x = amplitude * np.sum(wells * (1 - offsets**2 / well_width**2)) / well_width**2 + confinement
    curvature_y = amplitude * wells.sum() / well_width**2 + confinement
    return energy, gradient, curvature_x, curvature_y


def _well(amplitude, well_width, confinement):
    x = -1.0
    for _ in range(100):
        _, gradient, curvature, _ = _terms(x, amplitude, well_width, confinement)
        if curvature <= 0:
            raise ValueError("Gaussian well has no stable minimum")
        step = gradient / curvature
        x -= step
        if abs(step) < 1e-12:
            return x
    raise ValueError("Could not locate the well minimum")


def harmonic_tst(amplitude=3.0, kBT=0.4, mass=1.0, well_width=0.5, confinement=0.2):
    """Return harmonic-TST rate and barrier for the Gaussian double well."""
    if min(kBT, mass, amplitude, well_width, confinement) <= 0:
        raise ValueError("Thermal energy, mass, widths, amplitude and confinement must be positive")
    well_x = _well(amplitude, well_width, confinement)
    well_E, _, wx, wy = _terms(well_x, amplitude, well_width, confinement)
    saddle_E, _, sx, sy = _terms(0, amplitude, well_width, confinement)
    if sx >= 0 or sy <= 0:
        raise ValueError("Potential has no first-order saddle at x=0")
    barrier = saddle_E - well_E
    prefactor = np.sqrt(wx * wy / sy / mass) / (2 * np.pi)
    return float(prefactor * np.exp(-barrier / kBT)), float(barrier)


def _solve_amplitude(target, rate_target, parameters):
    fn = lambda amplitude: harmonic_tst(**(parameters | {"amplitude": amplitude}))[0 if rate_target else 1]
    lo, hi = 0.1, 1.0
    base = fn(lo)
    if (target > base if rate_target else target < base):
        raise ValueError("Target is outside the range of this Gaussian-well potential")
    while (fn(hi) > target if rate_target else fn(hi) < target):
        lo, hi = hi, hi * 2
        if hi > 1000:
            raise ValueError("Could not find a well amplitude for the requested target")
    for _ in range(60):
        mid = (lo + hi) / 2
        if (fn(mid) > target if rate_target else fn(mid) < target):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def well_amplitude_for_rate(target_rate, **parameters):
    """Return the Gaussian-well amplitude for a target harmonic-TST rate."""
    return _solve_amplitude(target_rate, True, parameters)


def _arrhenius(barrier, parameters):
    well_x = _well(parameters["amplitude"], parameters["well_width"],
                   parameters["confinement"])
    _, _, curvature, _ = _terms(well_x, parameters["amplitude"],
                                parameters["well_width"], parameters["confinement"])
    attempt = np.sqrt(curvature / parameters["mass"]) / (2 * np.pi)
    return float(attempt * np.exp(-barrier / parameters["kBT"]))


def _pmf_rate(path, bins, window):
    with np.load(path) as data:
        x, vx = data["positions"][:, 0], data["velocities"][:, 0]
    near = np.abs(x) <= window
    if bins < 1 or window <= 0 or not near.any() or not (x < 0).any():
        raise ValueError("PMF needs positive bins, saddle samples, and left-well samples")
    density, edges = np.histogram(x, bins=bins, density=True)
    saddle_bin = np.argmin(np.abs((edges[:-1] + edges[1:]) / 2))
    return float(np.mean(np.maximum(vx[near], 0)) * density[saddle_bin] / np.mean(x < 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("harmonic", "arrhenius", "pmf"), default="harmonic")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--target-transitions", type=float)
    choice.add_argument("--barrier-height", type=float)
    parser.add_argument("--production-time", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--data")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--barrier-window", type=float, default=0.1)
    parser.add_argument("--kBT", type=float, default=0.4)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--well-amplitude", type=float, default=3.0)
    parser.add_argument("--well-width", type=float, default=0.5)
    parser.add_argument("--confinement", type=float, default=0.2)
    args = parser.parse_args()

    if args.production_time is not None and args.n_steps is not None:
        parser.error("Choose --production-time or --n-steps")
    if args.production_time is not None:
        duration = args.production_time
    elif args.n_steps is not None:
        duration = (args.n_steps - args.burn_in) * args.dt
    else:
        duration = None
    if duration is not None and duration <= 0:
        parser.error("Production duration must be positive")

    p = dict(kBT=args.kBT, mass=args.mass,
             amplitude=args.well_amplitude, well_width=args.well_width,
             confinement=args.confinement)
    if args.method == "pmf":
        if not args.data:
            parser.error("--method pmf requires --data")
        rate = _pmf_rate(args.data, args.bins, args.barrier_window)
        barrier = None
    else:
        if args.target_transitions is None and args.barrier_height is None:
            parser.error("Choose --target-transitions or --barrier-height")
        if args.target_transitions is not None:
            if duration is None or args.target_transitions <= 0:
                parser.error("Target transitions require positive count and production duration")
            target_rate = args.target_transitions / duration
            if args.method == "harmonic":
                amplitude = well_amplitude_for_rate(target_rate, **p)
                rate, barrier = harmonic_tst(**(p | {"amplitude": amplitude}))
                print(f"Required Gaussian-well amplitude: {amplitude:.8g}")
            else:
                attempt = _arrhenius(0, p)
                if target_rate >= attempt:
                    parser.error("Target rate must be below the Arrhenius attempt frequency")
                barrier = args.kBT * np.log(attempt / target_rate)
                rate = _arrhenius(barrier, p)
            print(f"Required barrier height: {barrier:.8g}")
        else:
            barrier = args.barrier_height
            if barrier <= 0:
                parser.error("Barrier height must be positive")
            if args.method == "harmonic":
                amplitude = _solve_amplitude(barrier, False, p)
                rate, barrier = harmonic_tst(**(p | {"amplitude": amplitude}))
                print(f"Equivalent Gaussian-well amplitude: {amplitude:.8g}")
            else:
                rate = _arrhenius(barrier, p)

    print(f"Estimated rate: {rate:.8g} per MD time unit")
    print(f"Mean waiting time: {1 / rate:.8g} MD time units")
    if duration is not None:
        print(f"Expected transitions in production duration: {rate * duration:.8g}")


if __name__ == "__main__":
    main()
