## Objective

There are two technques for resolving subsurface density maps based on gravity measurements at surface. While solving for gravity from a density map is trivial, the inverse is an ill-posed problem without a unique solution.

## Inversion

Using Bayesian analysis techniques like [SIMPEGs inversion solvers](https://docs.simpeg.xyz/latest/content/user-guide/tutorials/03-gravity/index.html) one can resolve density contrasts based on gravity measurements. 

To test these solvers: Generate synthetic density contrast → Solve for gravity → Add noise to simulate realistic measurements → Solve for density contrast using SIMPEG (L2 or IRLS inverse solvers). 

[bayesian.ipynb](archive/bayesian.ipynb)

Unfortunately this method is unsuitable for non-linear / sparse problems amongst many shortcomings and is computationally inneficient.

## Deep Learning

A forward pass through a neural net may be more suitable at resolving non-linear / sparse features along with being more computationally efficient. However, to generate a neural net of this kind, one has to train the wieghts on synthetic data as there is limited subsurface density contrast maps.

Generate synthetic density contrast → Solve for gravity → Add noise to simulate realistic measurements → Train neural net

- density contrast (y)
- gravity measurement (x)

## Notes On Previous Work

#### Improved Gravity Inversion Method Based on Deep Learning with Physical Constraint and Its Application to the Airborne Gravity Data in East Antarctica
- CNN UNET (2D --> 3D)
- 20 000 samples
- 1g/cm^3 contrast
- 32 x 32 x 16 (1km cubes)

#### Deep Learning for 3-D Inversion of Gravity Data
- CNN UNET (2D --> 3D)
- 40 000 samples
- 0.1-1g/cm^3
- 3:1:1 ratio
- 64 x 64 x 32 (50m spacing)
- single block; single dipping slab; combined blocksl; combined dipping slabs

#### 3-D Gravity Inversion Based on Deep Convolution  Neural Networks
- CNN UNET (2D --> 3D Virtual (50 channels represent depth))
- 14 000 samples
- 3:1:1 ratio
- 112 x 112 x 50 channels (50m spacing)
- one, two, four prism

#### Three-dimensional gravity inversion  based on 3D U-Net++*
- CNN UNET (3D --> 3D)
- 12 000 samples
- 32 x 32 x 16
- 5:1 ratio
- difference from 3D UNET (plain skips vs dense/nested skips)

#### Three-Dimensional Gravity Inversion Based on Attention Feature Fusion
- CNN UNET (2D --> 3D)
- 32 x 32 x 16 (50m)
- 22 000 samples
- 1g/cm^3 density contrast

## Architecture (UNET 2D --> 3D)

Trying to replicate *Deep Learning for 3-D Inversion of Gravity Data* by *Zhang et al.*

<img src="documentation/zhang.png" alt="alt text" width="800" />

## Dataset 1 [[single_block.h5](datasets/single_block.h5)]

- 20 000 samples
- 4:1 training, validation [[single_block.npz](splits/single_block.npz)]
- 0-1 g/cm^3
- 32 x 32 x 16 (50m voxels)
- Randomly generate 1 block 0-30% of domain size within voxel grid
- flat topography
- noise: 0.05e-3 mGal, w/ 95% confidence
- Results: [[single_block](runs/single_block)]
    - $tensorboard --logdir=runs --bind_all

## Dataset 2

The following paper describes a method for generating much more plausible/realistic synthetic geology. More specifically, applying matrix transformations to reflect realisitc geological timelines to an inital deposition of material. The variance in this dataset is a function of Markov sampling of the following [Markov Matrix](StructuralGeo/src/geogen/generation/markov_matrix/default_markov_matrix.csv).

[Synthetic Geology -- Structural Geology Meets Deep Learning](https://arxiv.org/abs/2506.11164)

The generation of this dataset relies on a [forked version](https://github.com/kostubhagarwal/StructuralGeo) of the repo described in this paper. This fork adds a channel of information (in this case density) to 'Rock Category Mapping' as defined in StructuralGeo via a [lookup table](StructuralGeo/src/geogen/dataset/add_channel.py).

## To-do
- refactor gen/*
- refactor plot
- README.md
- Create a config system for:
    - Model hyperparameters (learning rates, architectures)
    - Data parameters (grid sizes, noise levels, density ranges)
    - Training settings (batch sizes, epochs, validation splits)
- Add Model Interpretability & Analysis Tools
    - Feature map visualization for encoder/decoder layers
    - Attention mechanisms to show which gravity measurements drive density predictions
    - Ablation studies on input channels (gravity vs height)
    - Analysis of failure modes on different geological scenarios

## Open Questions
### Geology
- find deposits vs determine geology?
- what kinds of deposits?
- domain size? 
- edge effects?
- depth?

### Deep Learning
- multi-sensor fusion?
- pre-train data analysis?
- what should i track during training?
- alternative architectures?
    - GAN / Flow Matching
    - NeRF
    - DeepSDF
    - FNO / DeepONet
    - PINN
- epoches, batches, etc?
- skip connections?





