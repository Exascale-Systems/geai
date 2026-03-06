## Objective

While, calculating the gravitational field from a known subsurface density map (forward solution) is trivial, although computationally expensive, the inverse solution is ill-posed (lacks a unique solution, solution is unstable, or there is no solution at all).

## Background

There are two main flavours to resolving a 3D subsurface density map.

1. Generating a guess (likely well-informed prior) for a 3D subsurface density map, forward modeling it, then comparing it with the measured 2D gravity map. 

2. Directly inverting the measured 2D gravity map to resolve a 3D subsurface density map. 

In industry, both these methods are used in conjuction to converge on a solution. 

[SIMPEG](https://docs.simpeg.xyz/latest/content/user-guide/tutorials/03-gravity/index.html) is an open source library enabling both forward modelling and the direct inversion (least squares minimization, sparse norm, etc) of a 2D gravity measurement to resolve a 3D subsurface density map. 

SIMPEG's inversion solvers are classified as deterministic optimization methods. Unfortunately, the nature of the problem

- non-linear
- sparse
- long wavelengths

makes these methods somewhat inneffective at resolving accurate 3D subsurface density maps. 

To evaluate the process I: 

Generated a synthetic density contrast (Cube of 2.0g/cm) → Calculated gravity at surface → Add noise to surface gravity measurement (to simulate realistic survey) → Solve for density contrast using SIMPEG (L2 or IRLS inverse solvers). 

[bayesian.ipynb](archive/bayesian.ipynb)

The results tend to severly smooth out results and weight the voxels with less depth far more than those are further wavelengths. While some people adjust the Jacobian sensitivity matrix (depth-weigthing, etc) this method is still limited.

On the otherhand, there are stochastic sampling methods like MCMC. While this can give very good solutions, they are extremely computationally expensive. 

## Deep Learning

Our thesis is that a forward pass through a neural net may be more suitable at directly inverting a 2D gravity measurement while being able to resolve non-linear, sparse, long wavelength 3D subsurface density maps. This has the added benefit of frontloading computation during pretraining with inference being extremely lightweight. 

One has to train the weights of the neural net via deep learning to perform the inversion. Since there is limited subsurface density maps, one must use synthetic data. The training process requires the following:

Generate synthetic subsurface density map → Calculate gravity at surface → Add noise to surface gravity measurement (to simulate realistic survey) → Forward pass the noisy surface gravity measurement → Calculate loss between predicted 3D subsurface density map and actual data (the label) → Backprop through the network:

- gravity measurement (x)
- density contrast (y)

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

## Status Quo

1. Manually model a 3D subsurface density map based on geochemistry, known geology, other priors, etc
2. Forward model the 2D gravity map at surface
3. Compare with actual measurement

in parallel:

1. Directly invert 2D gravity map at surface
2. Compare predicited density map with manual modelled 3D subsurface density map

repeat.

eg: https://www.youtube.com/watch?v=FwN9O1AnS3g&t=764s (mira & xcalibur)

## The End-game:

1. Generate synthetic dataset of 3D subsurface geology maps
2. Forward model (SIMPEG or Neural Operator) to solve for 2D gravity map
3. Train a flow-matching model (stochastic) to generate a distribution of plausible geologies
4. Forward model and compare residuals of 2D gravity map. Verify against more realworld data, repeat.

These guys have spelled it out: https://arxiv.org/pdf/2506.11164

## To-do

Once we understand how to maximize the performance of a U-Net we will flow-match.

## Open Questions

1. is a neural net just memorizing shit? can it actually work out of distribution and respect physics?
2. can we generate realistic enough training data?
3. can we train a neural net on targeted training data? (embed priors into synthetic data generation)
4. edge effects?
5. survey properties (spacing, noise)
6. physics limitations (depth, feature size, etc)

#### High Priority

- Dataset 1
    - loss function
        - MAE
        - dice
        - IoU
        - MSE (regression) & Dice/IoU (segmentation)
        - simpeg forward pass - compare true gravity to predicted gravity
    - data augmentation (pre-training)
        - rotation
        - flips
        - translation
    - data augmentation validation (post-training)
        - rotation
        - flips
        - translation
    - weight decay
    - learning rate
    - batch size
    - normalization
    - dropout
    - skip modules
    - attention
    - 3D --> 3D UNET?

- Dataset 2
    - do the same, hopefully we understand nn's more

- Real data
    - UTAH Forge
    - Los Angeles Basin
    - Otway Basin

- Flowmatching

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
- deeplearning + simpeg regularization hybrid
- Generate clean dataset with vector and grad components
- Train & Evaluate (0.5, 5, 50, 500 uGal nn)
- Refactor ()
- Train & Evaluate (vector vs gz vs gzz vs gz & gzz vs vector & gzz)
- Train & Evaluate (dl vs dl+simpeg vs simpeg)

## Notes

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

#### Mark McLean '3D inversion modelling of Full Spectrum FALCON® airborne gravity data over Otway Basin'
- https://www.youtube.com/watch?v=FwN9O1AnS3g&t=764s (mira & xcalibur)
    - terrain correction
    - dtu15 free-air satellite dataset
    - potential field models