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

## Goals
1. What economic value can be resolved?
2. Classical survey vs exascale survey?
3. How to improve predictions?

## To-do
- train.py
    - normalization
    - loss function
    - data augmentation
        - noise
        - rotation
        - flips
    - weight decay
    - learning Rate
    - batch size
- noise vs error (l1, rmse, iou, dice)
    - Deeplearning (no, medium, high)
    - SIMPEG
- scalar vs vector measurement
- StructuralGeo dataset
------
- refactor plot.py
- nn.py
    - dropout
    - skip modules
    - attention
    - 3D --> 3D UNET?
    - multi-sensor fusion
        - Ablation study
- multi gpu
- confidence

## Completed
- inspect.py / plot.py
    - gravity
        - residual plot
            - RMSE
            - R^2
    - density map
        - mse vs l1 vs dice vs IoU
        - slice plot (x,y,z)
        - slice residuals plot (x,y,z)
            - mse vs l1 vs IoU vs dice coeff.
- train.py
    - track
        - gradient norms
        - weight histogram drift
        - *feature map visualization
- figure out gal/sqrt(Hz)
- heavy refactor
- SIMPEG eval.py
- consolidate/refactor metrics.py, eval.py, sample.py, data.py
- consolidate/refactor sample.py, eval.py
- refactor eval.py

## Open Questions
### Geology
- edge effects?
- markov matrix?
- density mask?
- what kinds of geophysical characteristics?
- domain size? 
- depth?

### Deep Learning
- alternative architectures?
    - Flow Matching / GANs
    - NeRF
    - DeepSDF
    - FNO / DeepONet
    - PINN