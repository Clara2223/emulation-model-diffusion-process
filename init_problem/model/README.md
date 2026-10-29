# Model package

 Run from `init_problem`:

```bash
../special_energy/bin/python -m model.run --epochs 1000 --max-samples 50000
```

The top-level script is a launcher. Everything used by the model is in this folder:

- `data.py`: loads positions, velocities, and energies.
- `model_2D.py`: small score and velocity networks for 2D position samples.
- `model_3D.py`: score and velocity networks for 3D samples and transitions.
- `train.py`: trains the 2D models and records their losses.
- `predict.py`: generates samples, calculates diagnostics, and makes plots.
- `run.py`: connects the files and runs the experiment.
- `run_3D.py`: runs the 3D joint-sample and transition experiments.

Both models learn the 2D equilibrium distribution of positions `(x, y)` only. Neither receives physical MD time, velocity, or energy. Diffusion also receives its noise level and predicts a score; flow receives its internal transport coordinate and predicts a velocity. These internal coordinates are part of the generative algorithms, not physical trajectory time. Both use the same plots and diagnostics, so no separate prediction script is needed. Generated samples are saved in `init_problem/predictions/`.

All model paths use the shared `DEVICE` selected in `model_2D.py`. Training uses direct PyTorch optimizer loops on that device. The 2D runner keeps its dataset CPU-backed and moves each batch to the selected device; the 3D runner creates transition tensors as `float32` on that device. NumPy-backed loading uses `torch.from_numpy` where CPU storage is intentional.

Prediction figures overlay MD and generated position densities and potential-energy distributions. The generated-data legend identifies the selected model as either diffusion or flow.

The plotted potential is read from the dataset metadata. Older archives without this metadata are identified from their filename (`chain` or `double_well`); unrecognized names raise an error instead of silently plotting the wrong potential. H-barrier plots use the saved ridge amplitude and width.

For chain archives, both runners map only x to `(x % period) / period` before normalization and noise is added. Set the period with `--period` (default 32); y and physical MD time are unchanged. Generated x values are mapped back to the physical period, and logs identify this input treatment as `fractional_x_v1` and record the selected period. The models' internal diffusion/flow coordinate is named `tau`; `t` refers to physical MD time.

Each new prediction-log entry also records mean marginal and joint histogram Hellinger distances between reference and generated samples. Both range from 0 (matching histograms) to 1 (maximally different); the CSV begins with brief metric definitions. These are logged metrics only and do not alter the existing 2D plots. Sampler settings are included in the log and output filename: diffusion records noise levels and steps per level, while flow records ODE integration steps.

Both `--steps-per-level` and `--ode-steps` default to 50. Diffusion applies that many stochastic score-based correction updates at each noise level (600 updates total with the default 12 noise levels); flow uses that many numerical integration steps to follow its learned velocity field from noise to samples. Neither is a physical MD-time step.

## 3D samples and trajectory transitions

The separate 3D runner requires an archive with `positions` and `md_time` arrays:

```bash
python -m model.run_3D --data MD/train_data/3D_double_well_data.npz --objective joint
python -m model.run_3D --data MD/train_data/3D_double_well_data.npz --objective transitions --model flow
```

`--objective joint` trains diffusion or flow on shuffled `(x, y, physical MD time)` samples. It learns their joint distribution, not temporal order. `--objective transitions` instead trains a conditional distribution of the next `(x, y)` from the current `(x, y, physical MD time)` using consecutive saved frames, then rolls it forward to estimate basin-crossing rates. The transition objective uses the saved chronology; the joint objective does not. Its crossing-rate estimate counts basin-label changes per unit physical time (two basins split at `x=0` for double wells; 16 periodic bins for the chain), so it is a sampled discrete-time diagnostic, not a guaranteed kinetic rate. Generated arrays, a task-specific comparison figure, and training loss are saved separately. Non-chain 2D behavior remains unchanged.

Generate a high-barrier double-well trajectory with:

```bash
python MD/MD_sim.py --potential double-well-Hbarrier
```

This adds a Gaussian ridge along `x=0`, with default width `0.25`. Its amplitude is solved using harmonic transition-state theory so the expected number of crossings in the first half of the saved trajectory is one. The target rate is `k = 1 / (T_saved / 2)`. The calculation uses the two-dimensional harmonic TST prefactor (including the stable transverse frequency at the saddle); for the default `100,000,000` steps, `dt=0.005`, `50,000` burn-in steps, and `kBT=0.4`, it gives a barrier of about `4.99` energy units above the well and a ridge amplitude about `2.90`. These are TST estimates, not a guarantee of exactly one observed transition: friction and recrossings can change the realized count. Override the defaults with `--target-transitions-half` and `--barrier-width`.

Training reserves a deterministic 10% of the selected samples for validation. Diffusion training samples `tau` uniformly from `[1e-6, 1)` and sets `sigma_tau = sigma_min * (sigma_max / sigma_min)**tau`, with `sigma_min=0.02` and `sigma_max=1`. It then forms `X_tau = Z + sigma_tau * E`; `sigma_tau` is the noise scale used by the score model. The 2D trainer reuses fixed validation noise and `tau` so its curve is comparable across epochs. The loss plot contains both training and validation loss by epoch. Diffusion generation separately uses a descending discrete sigma schedule for its Euler-Maruyama Langevin updates.

The 2D trainer is intentionally epoch-based: for each epoch it updates the model on every training batch, measures average training loss, then measures validation loss without updating the model. It returns the model and the two loss histories to `run.py`.

For a quick local run:

```bash
../special_energy/bin/python -m model.run --epochs 500 --max-samples 10000
```

Select flow matching with `--model flow`; the default is `--model diffusion`.
Training checkpoints are not saved by default. Add `--save-checkpoint` to either
runner to save the model weights, optimizer state, normalization values, run
settings, and loss histories as a small `.pt` file in `init_problem/predictions/`.
The checkpoint filename includes the run settings and timestamp.
Resume training by passing the checkpoint to either runner:

```bash
python -m model.run --resume-checkpoint path/to/checkpoint.pt --epochs 100 --save-checkpoint
python -m model.run_3D --resume-checkpoint path/to/checkpoint.pt --epochs 100 --save-checkpoint
```

The checkpoint supplies the original data path and model/training settings; `--epochs`
is the number of additional epochs. The resumed run writes a new checkpoint when
`--save-checkpoint` is given. If the saved data path is unavailable, pass `--data`.

Both networks are deliberately small MLPs. The diffusion implementation is a compact score-based baseline, not a full DDPM/SDE solver. Einstein diffusion and the Kramers rate remain MD-trajectory diagnostics; they are not predicted by either 2D model.

Each PNG in `init_problem/predictions/` is registered in `prediction_log.csv` with its model version, Einstein diffusion, Kramers barrier height, Kramers transition rate, and device.
