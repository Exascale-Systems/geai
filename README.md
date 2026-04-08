# Generative Exploration AI (GEAI)

## Background

Remote sensing of the subsurface is fundamentally an ambiguous problem. Interpolating a 3D volume from a 2D measurement is inherently ill-posed (non-unique, unstable, underdetermined (2D → 3D)) — meaning there are many possible 3D geological models that can produce the same set of 2D measurements.

As such, there are two main flavours to resolving a 3D subsurface density map in industry today:

1. **Model → simulate → compare** — propose a geological model, forward simulate, compare with observed data, repeat.
2. **Direct inversion** — invert measurements into a model, often heavily regularized.

In industry, both these methods are used in conjunction to converge on a solution.

SIMPEG is an open source library enabling both forward modelling and the direct inversion (least squares minimization, sparse norm, etc) of a 2D gravity measurement to resolve a 3D subsurface density map. SIMPEG's inversion solvers are classified as deterministic optimization methods. Unfortunately, the nature of the problem:

- non-linear
- sparse
- long wavelengths

makes these methods susceptible to producing severely smoothed results that over-weight shallow voxels. The gravity forward problem is linear (`g = Jρ`), so SimPEG can solve it as a linear least-squares problem — but the sensitivity matrix **J** decays sharply with depth, making the inversion inherently depth-ambiguous and sensitive to regularization choice. While depth-weighting the Jacobian partially compensates, it doesn't resolve the fundamental non-uniqueness.

Our thought was then that a neural net would be better suited at directly inverting a 2D gravity measurement while being able to resolve non-linear, sparse, long wavelength 3D subsurface density maps. This has the added benefit of frontloading computation during pretraining with inference being extremely lightweight.

This would look like:

```
Generate geology → Compute surface gravity → Add noise (simulate survey)
    → Forward pass → Loss(predicted density map, ground truth) → Backprop
```

- `x`: gravity measurement
- `y`: density contrast

However, this does not solve the fundamental problem. Even with a perfect solver that produces realistic subsurface models, the inverse problem remains non-unique—there are many models that fit the same data. As a result, service providers typically deliver a model that fits the observations, but not necessarily one that is correct or representative of the full solution space.

This makes **stochastic sampling approaches** more appropriate, as they aim to approximate the full posterior ( P(m \mid d) ) rather than a single solution. Traditional methods like MCMC can achieve this, but are computationally expensive. Unfortunately, this workflow is unsuited to humans because:

* **The solution space is combinatorial and non-unique:** the number of valid models ( m \in \mathbb{R}^N ) that explain ( d ) is effectively exponential in (N). Humans can only evaluate a handful; the system can approximate ( P(m \mid d) ) over this space.

* **Evaluation is cheap but search is the bottleneck:** forward physics ( g(m) ) is well-defined, so scoring ( P(d \mid m) ) is straightforward—the hard part is exploring enough candidates. This is exactly where computation dominates humans.

* **The problem is sequential, not static:** optimal exploration requires planning over future measurements, i.e. maximizing expected value over action sequences. Humans reason myopically; the system can simulate and optimize over many possible futures.

**Generative models (flow matching, diffusion)** address this by learning the distribution upfront, shifting computation to training and enabling fast sampling at inference.

Finally, since **drilling cores provide ground truth**, the problem naturally becomes a **sequential, data-driven process**, making it well-suited to a computational pipeline that continuously samples, updates, and refines beliefs as new measurements are acquired.

This problem is not limited by physics or modeling — it is limited by *search over uncertainty*, which is precisely where computation dominates human reasoning. This results in a much more cost-effective, efficient exploration process.

## Vision

Build an agent to drive end-to-end geophysical exploration — analogous to AlphaGo.

```
belief -> action -> possible observation -> plan -> measurement -> updated belief -> repeat
```

1. Generate geology
2. Suggest measurements (where most uncertain)
3. Take measurements (gravity, drill, mag, etc.)
4. Update geology
5. Repeat until satisfied

**Bayesian formulation:**
```
P(m|d) = P(m) * P(d|m)

New Belief = Old Belief * New Measurement Likelihood

m: structural geology
d: observations

P(m):   structural geology prior
P(d|m): likelihood of measurement given structural geology prior
P(m|d): structural geology given measurement
```

**Steps:**

1. Generate many samples of synthetic geology (Flow Matching, GANs)

2. Planning (MCTS or POMDP):
```
for candidate action a across all samples:
    simulate possible d ~ P(d|m, a)
        - predicted measurement value * each sample's voxel likelihood
    update belief
    evaluate downstream value
```

3. Choose highest expected value measurement (e.g., drill: angle, depth, x/y location)

4. Take a real measurement

5. Update belief via conditional generative geology

### Next Steps

1. Posterior Flow Sampling Implementation of Gravity (https://github.com/chipnbits/flowtrain_stochastic_interpolation)

2. Conditional flow matching gravity (https://github.com/chipnbits/flowtrain_stochastic_interpolation)

3. Confidence of each voxel (feature extent)
Feature extent & Model scoring: https://github.com/chipnbits/flowtrain_stochastic_interpolation

4. Planning engine (start with drilling): For a given block with a given uncertainty, determine best drill (angle, depth, (x,y) location)

### Infrastructure & Links

- [Datasets & models](https://drive.google.com/file/d/1VrqcjQ8eliTs9gD75zIvI0jUpR37lg5K/view?usp=sharing) — Google Drive
- [StructuralGeo fork](https://github.com/kostubhagarwal/StructuralGeo) — realistic geology data generation
- [flowtrain_stochastic_interpolation](https://github.com/kostubhagarwal/flowtrain_stochastic_interpolation) — template for posterior flow sampling

---

## Repo

```
scripts/
  gen.py                entry point: generate dataset (blocks / structuralgeo)
  train.py              entry point: train the model
  eval.py               entry point: evaluate (nn / hybrid / bayesian)
  raw.py                visualize a single raw sample from an HDF5 file

src/
  data/
    dataset.py          HDF5 streaming dataset, component extraction, normalization
    transforms.py       noise injection, norm/denorm

  gen/
    gen.py              mesh, topography, random blocks, gravity survey (SimPEG)
    batch.py            batch generation — blocks dataset
    hdf5_writer.py      writes samples to HDF5
    structuralgeo/
      gen.py            StructuralGeo integration (realistic geology)
      batch.py          batch generation — structuralgeo dataset

  nn/
    unet.py             GravInvNet — 2D→3D UNet
    engine.py           training loop, checkpointing, TensorBoard logging
    loss_functions.py   DiceLoss

  evaluation/
    nn.py               NN evaluation + single-sample visualization
    hybrid.py           NN prediction used as SimPEG initial model, then inverted
    simpeg.py           pure Bayesian inversion (SimPEG)
    metrics.py          TorchMetrics, NumpyMetrics (RMSE, L1, IoU, Dice)
    plotter.py          gravity and density visualizations

params.yaml             all hyperparameters (data, gen, train, eval)
dvc.yaml                pipeline stage definitions
```

## Data Format

### HDF5 schema (`data/<ds_name>.h5`)

```
globals/
  hx            float32[nx]       cell widths — x axis
  hy            float32[ny]       cell widths — y axis
  hz            float32[nz]       cell widths — z axis
  shape_cells   int32[3]          (nx, ny, nz)

samples/<seed>/
  gravity_data        float32[n_components × n_receivers]  flattened, component-major
  receiver_locations  float32[n_receivers, 3]              (x, y, z) survey stations
  true_model          float32[n_active]                    density contrast (g/cc) for active cells only
  ind_active          uint8[n_total_cells]                 boolean mask — active (subsurface) cells
```

### Gravity components

9 components available; select any subset via `train.components`:

```
gx  gy  gz          first-order (mGal)
gxx gxy gxz
gyy gyz gzz         second-order / gravity gradient tensor (Eötvös)
```

Each selected component adds one input channel to the model.

### Splits (`splits/<split_name>.npz`)

```
tr    int[n_train]   training indices (80%)
va    int[n_val]     validation indices (20%)
```

Auto-generated on first run with `seed=0` and saved. Reused on subsequent runs if the file exists. Referenced by `data.split_name` in `params.yaml`.

---

## Model

**GravInvNet** — 2D gravity map → 3D density volume

```
Input:   (B, C, ny, nx)      C = number of gravity components, ny=nx=32
Output:  (B, nz, ny, nx)     nz=16
```

Architecture: 2D encoder → 1×1 conv dimension bridge → 3D decoder

```
Encoder2D   Conv2d stride-2 ×4    channels: in → 128 → 256 → 512 → 1024
                                  ResBlock2D (BN → conv → BN → LeakyReLU(0.1)) at each stage
DimTransform 1×1 Conv2d + Conv3d  injects depth dim: (B,1024,H,W) → (B,1024,1,H,W)
Decoder3D   ConvTranspose3d ×4    channels: 1024 → 512 → 256 → 128 → 64 → 1
                                  BN + LeakyReLU(0.1) after each up-conv
Head        Conv3d 1×1            projects to density scalar, squeezes channel dim
```

**Training:**
- Optimizer: Adam (`lr`, `wd`)
- Grad clip: 1.0 (L2 norm)
- AMP: `GradScaler` (mixed precision)
- Early stop: val loss < `min_loss`
- Saves `checkpoints/best.pt` (best val loss, full state) and `checkpoints/<model_name>_final.pt` (last epoch)
- Resumes from `checkpoints/best.pt` if it exists — restores model weights, optimizer state, scaler, and epoch count automatically (re-run `train.py` to continue)
- Eval loads from `models/<model_name>.pt` — manually promote a checkpoint when ready:
  ```sh
  cp checkpoints/best.pt models/my_model.pt
  ```

**Loss functions** (`train.experiments.<model_name>.loss_function`):

| Key | Formula |
|---|---|
| `mse` | `mean((pred - target)²)` |
| `dice` | `1 − (2·∑pᵢtᵢ + ε) / (∑pᵢ + ∑tᵢ + ε)`, inputs mapped to [0,1] first |

**Noise augmentation** (`train.noise.accuracy`, `train.confidence`):

Gaussian noise added to gravity input during training to simulate instrument uncertainty:
```
σ = accuracy / ppf((1 + confidence) / 2)
```
E.g. `accuracy=0.05, confidence=0.95` → `σ ≈ 0.026 mGal`

---

## Dependencies

- **[SimPEG](https://simpeg.xyz)** — forward modeling and inversion. Generates ground truth gravity data and serves as a physics-based regularizer in the hybrid eval pipeline.
- **[StructuralGeo](https://github.com/kostubhagarwal/StructuralGeo)** (forked) — realistic synthetic geology via Markov-sampled structural history. Installed from the fork via `uv`.
- **[DVC](https://dvc.org)** — tracks datasets, models, splits, checkpoints. Pipeline in `dvc.yaml`, parameters in `params.yaml`.
- **[TensorBoard](https://tensorboard.dev)** — loss curves, gradient norms, weight histograms.

## Setup

```sh
uv sync
```

Datasets and model checkpoints are not included in the repo. Download them from [Google Drive](https://drive.google.com/file/d/1VrqcjQ8eliTs9gD75zIvI0jUpR37lg5K/view?usp=sharing) and place `.h5` files in `data/` and `.pt` files in `models/`. Alternatively, regenerate from scratch with `dvc repro gen_data`.

---

## Workflows

All parameters live in `params.yaml`. Each script reads from it by default; CLI flags override individual values for quick iteration without editing the file.

---

### 1. Generate a dataset

Set `gen.generator` to choose the model type, then run:

```sh
python scripts/gen.py
```

**`params.yaml` keys:**

```yaml
gen:
  generator: blocks        # blocks | structuralgeo
  out_path: data/foo.h5   # output file
  ds_size: 5000           # number of samples

  # blocks only
  x_dom: 1600.0
  y_dom: 1600.0
  z_dom: 800.0
  n_xy: 32
  n_z: 16
  n_blocks: 1
  size_frac_min: 0.10
  size_frac_max: 0.30
  density_min: 0.0
  density_max: 1.0

  # structuralgeo only
  bounds: [[0, 6.4e4], [0, 6.4e4], [0, 3.2e4]]
  resolution: [32, 32, 16]
```

**CLI overrides** (no need to edit the file for one-off runs):

```sh
python scripts/gen.py --generator blocks --out-path data/blocks_test.h5 --ds-size 100
python scripts/gen.py --generator structuralgeo --out-path data/sgeo_test.h5 --ds-size 50
```

---

### 2. View a dataset sample

Interactively visualize a single sample from an HDF5 file — opens a 3D density plot with a movable clipping plane and gravity anomaly maps:

```sh
python scripts/raw.py data/foo.h5             # first sample (idx=0)
python scripts/raw.py data/foo.h5 --idx 12    # specific sample
```

---

### 3. Train the neural network

Set the target model name and training config, then run:

```sh
python scripts/train.py
```

**`params.yaml` keys:**

```yaml
data:
  ds_name: single_block_v2    # loads data/<ds_name>.h5
  split_name: single_block_v2 # loads splits/<split_name>.npz

train:
  model_name: my_model        # saves to checkpoints/my_model_final.pt + checkpoints/best.pt
  device: cpu                 # cpu | cuda
  batch_size: 32
  n_samples: null             # limit training to N samples (null = all)
  train_split: 0.8            # train/val fraction when generating new splits
  lr: 1e-3
  wd: 0.0
  max_epochs: 100
  eval_interval: 5            # validate every N epochs
  components: [gz]            # gravity components used as input: gx | gy | gz
  noise:
    accuracy: 0.05            # noise level injected during training
  confidence: 0.95
  min_loss: 1e-6              # early-stop: halt when val loss drops below this
  experiments:
    my_model:
      loss_function: mse      # mse | dice
```

`experiments` is a dict keyed by model name — each entry configures that model's loss function. Multiple models can coexist; the active one is selected by `train.model_name`.

**CLI overrides:**

```sh
python scripts/train.py --model my_model --epochs 50 --bs 64 --device cuda
python scripts/train.py --ds-name structuralgeo_v1
```

Checkpoints are written to `checkpoints/<model_name>_final.pt` (last epoch) and `checkpoints/best.pt` (best val loss). Loss curves and metrics stream to TensorBoard:

```sh
tensorboard --logdir=logs
```

---

### 4. Evaluate

Set `eval.mode` to choose the evaluation pipeline:

| Mode | Description |
|---|---|
| `nn` | Forward pass through the trained NN only |
| `hybrid` | NN prediction used as the SimPEG initial model, then refined by inversion |
| `bayesian` | Pure SimPEG inversion from scratch (no NN) |

**`params.yaml` keys:**

```yaml
eval:
  model_name: my_model   # loads models/<model_name>.pt
  run_name: null         # metrics filename: eval_{run_name}.json (null = model_name)
  mode: nn               # nn | hybrid | bayesian
  split: va              # tr | va — which split to evaluate
  output_dir: metrics    # where to write the JSON metrics file
  threshold: 0.1         # binarization threshold for IoU / Dice
  noise:
    accuracy: 0.05       # noise level applied to eval data
  confidence: 0.95
  inversion:             # SimPEG GNCG settings (bayesian + hybrid)
    max_iter: 50
    max_iter_ls: 20
    lower: -1.0          # density contrast bounds (g/cc)
    upper: 1.0
    cg_maxiter: 10
    cg_atol: 1.0e-4
    beta0_ratio: 10.0
    cooling_factor: 5.0
    cooling_rate: 1
    chifact: 1.0         # target misfit (1 = fit to noise)
  hybrid:
    alpha_s: 1.0         # smallness weight anchoring inversion to NN reference model
```

**Run the full eval set** (headless, writes metrics JSON):

```sh
python scripts/eval.py
```

**Evaluate a single sample with visualizations** (3D density plots, slice residuals, gravity maps):

```sh
python scripts/eval.py --idx 0                         # sample 0, mode from params.yaml
python scripts/eval.py --idx 5 --mode hybrid           # override mode
python scripts/eval.py --idx 0 --mode bayesian         # pure SimPEG, no model needed
```

**Other CLI overrides:**

```sh
python scripts/eval.py --mode nn --split tr            # evaluate on training split
python scripts/eval.py --model best --headless         # different checkpoint, suppress plots
```

Metrics are printed to stdout and saved to `metrics/eval_<run_name>.json` (`run_name` defaults to `model_name`).

---

## Metrics

All evaluation modes output the same four metrics, written to `metrics/eval_<run_name>.json`:

| Metric | Formula | What it measures |
|---|---|---|
| **RMSE** | √(mean((pred − true)²)) | Overall density contrast error in g/cc — lower is better |
| **L1** | mean(\|pred − true\|) | Mean absolute error — less sensitive to outliers than RMSE |
| **IoU** | \|pred ∩ true\| / \|pred ∪ true\| | Spatial overlap of predicted vs true anomaly after binarization at `threshold` |
| **Dice** | 2·\|pred ∩ true\| / (\|pred\| + \|true\|) | Same as F1 — rewards partial overlap more than IoU; preferred when anomalies are small |

Binarization: a voxel is considered "anomalous" if its predicted density contrast exceeds `eval.threshold` (default `0.1` g/cc). IoU and Dice operate on these binary volumes.

RMSE and L1 operate on the raw (continuous) predicted density values before binarization.

---

## DVC

DVC tracks three pipeline stages: `gen_data → train → eval`. All hyperparameters live in `params.yaml`. DVC detects when params or dependencies change and only re-runs what's stale — it won't regenerate a 20K-sample dataset just because you tweaked `lr`.

### params.yaml

Central config file for all pipeline parameters, grouped by stage:

| Section | Controls |
|---|---|
| `data` | dataset name, split name |
| `gen` | generator choice, domain size, mesh resolution, dataset size |
| `train` | model name, batch size, n_samples, train_split, lr, epochs, loss function, noise, components |
| `eval` | model name, run name, mode, threshold, noise, inversion hyperparams, hybrid alpha_s |

DVC reads this file to detect changes — if a param listed in `dvc.yaml` changes, the dependent stage is marked stale and will re-run on the next `dvc repro`.

### dvc.yaml

Defines the pipeline stages. Each stage has:

- **`cmd`** — the shell command to run
- **`params`** — which `params.yaml` keys this stage depends on (change → re-run)
- **`deps`** — file dependencies (e.g. the `.h5` file that `train` needs from `gen_data`)
- **`outs`** — output files to cache (DVC stores these by content hash in `.dvc/cache/`)
- **`metrics`** — output JSON files to track with `dvc metrics show`

Stages are linked by deps/outs — `train` depends on the file that `gen_data` produces, so DVC knows the order automatically.

### Run the full pipeline

```sh
dvc repro
```

### Run only one stage

```sh
dvc repro gen_data   # generate dataset only
dvc repro train      # train only (skips gen_data if data is unchanged)
dvc repro eval       # evaluate only (skips train if checkpoint is unchanged)
```

DVC checks each stage's dependencies first. If `data/single_block_v2.h5` already exists and `gen.*` params haven't changed, `dvc repro train` will skip gen_data entirely and go straight to training.

### Freeze a stage permanently

If you never want DVC to re-run data generation (e.g. dataset is fixed), freeze it:

```sh
dvc freeze gen_data     # dvc repro will always skip this stage
dvc unfreeze gen_data   # re-enable it
```

### Inspect results

```sh
dvc metrics show        # print all tracked metrics (train + eval JSON)
dvc params diff         # show what params changed since last run
dvc dag                 # visualize the pipeline graph
```

### Change an experiment

Edit `params.yaml` (e.g. change `train.model_name`, `train.lr`, `train.experiments`), then:

```sh
dvc repro train eval    # re-run only the affected stages
```

---

## TensorBoard

Training logs loss curves, gradient norms, weight histograms, and eval metrics in real time.

```sh
tensorboard --logdir=logs --bind_all
```

Open `http://localhost:6006` in your browser.

### What's logged

| Tab | What you see |
|---|---|
| **Scalars / Loss** | `Loss/train` and `Loss/val` per epoch |
| **Scalars / Metrics** | RMSE, L1, IoU, Dice on the validation set (logged every `eval_interval` epochs) |
| **Scalars / Hyperparams** | LR and weight decay (useful when using schedulers) |
| **Scalars / Gradients** | Gradient norm per batch — watch for spikes (exploding) or collapse to zero (vanishing) |
| **Histograms / Weights** | Weight distribution drift per layer per epoch — useful for detecting dead neurons or saturation |

## Resources

See [`resources/`](resources/) for collected research papers.
