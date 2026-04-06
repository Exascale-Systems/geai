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

makes these methods somewhat ineffective at resolving accurate 3D subsurface density maps. The results tend to severely smooth out results and weight the voxels with less depth far more than those at further wavelengths. While some people adjust the Jacobian sensitivity matrix (depth-weighting, etc.) this method is still limited.

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
  gen.py                entry point: generate dataset
  train.py              entry point: train the model
  eval.py               entry point: run evaluation

src/
  data/
    dataset.py          HDF5 streaming dataset, component extraction, normalization
    transforms.py       noise injection, norm/denorm

  gen/
    core.py             mesh, topography, random blocks, gravity survey (SimPEG)
    batch.py            batch generation — dataset 1 (random blocks)
    hdf5_writer.py      writes samples to HDF5
    structuralgeo/
      gen.py            StructuralGeo integration (realistic geology)
      batch.py          batch generation — dataset 2

  models/
    unet.py             GravInvNet — 2D→3D UNet

  train/
    engine.py           training loop, checkpointing, TensorBoard logging
    loss_functions.py   DiceLoss

  evaluation/
    pipeline.py         orchestrator: selects nn / bayesian / hybrid
    nn.py               NN evaluation + single-sample visualization
    hybrid.py           NN prediction as SimPEG initial model
    simpeg.py           pure Bayesian inversion (SimPEG)
    metrics.py          TorchMetrics, NumpyMetrics (RMSE, L1, IoU, Dice)
    plotter.py          gravity and density visualizations

params.yaml             all hyperparameters (data, gen, train, eval)
dvc.yaml                pipeline stage definitions
```

## Dependencies

- **[SimPEG](https://simpeg.xyz)** — forward modeling and inversion. Generates ground truth gravity data and serves as a physics-based regularizer in the hybrid eval pipeline.
- **[StructuralGeo](https://github.com/kostubhagarwal/StructuralGeo)** (forked) — realistic synthetic geology via Markov-sampled structural history. Installed from the fork via `uv`.
- **[DVC](https://dvc.org)** — tracks datasets, models, splits, checkpoints. Pipeline in `dvc.yaml`, parameters in `params.yaml`.
- **[TensorBoard](https://tensorboard.dev)** — loss curves, gradient norms, weight histograms.

## Setup

```sh
uv sync
```

Datasets and model checkpoints are not included in the repo. Download them from [Google Drive](https://drive.google.com/file/d/1VrqcjQ8eliTs9gD75zIvI0jUpR37lg5K/view?usp=sharing) and place `.h5` files in `data/` and `.pt` files in `checkpoints/`. Alternatively, regenerate from scratch with `dvc repro gen_data`.

---

## DVC

DVC tracks three pipeline stages: `gen_data → train → eval`. All hyperparameters live in `params.yaml`. DVC detects when params or dependencies change and only re-runs what's stale — it won't regenerate a 20K-sample dataset just because you tweaked `lr`.

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

### Typical healthy run

- `Loss/train` and `Loss/val` both decrease and track each other closely (no large gap = no overfitting)
- Gradient norm stays in the 0.01–1.0 range
- Weight histograms shift gradually without collapsing to zero

## Resources

See [`resources/`](resources/) for collected research papers.
