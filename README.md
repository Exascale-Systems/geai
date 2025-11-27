## Objective

There are two technques for resolving a subsurface density map from surface gravity measurements. While, calculating the gravity from a known subsurface map is trivial, the inverse is an ill-posed problem without a unique solution.

## Inversion

Bayesian analysis techniques like [SIMPEG](https://docs.simpeg.xyz/latest/content/user-guide/tutorials/03-gravity/index.html) enable the resolution of  subsurface density contrasts. 

Evaluation of the process requires the following: 

Generate synthetic density contrast → Calculate gravity at surface → Add noise to surface gravity measurement (to simulate realistic survey) → Solve for density contrast using SIMPEG (L2 or IRLS inverse solvers). 

[bayesian.ipynb](archive/bayesian.ipynb)

Unfortunately this method is unsuitable for non-linear / sparse problems, struggles at long wavelengths, and is computationally expensive.

## Deep Learning

A forward pass through a neural net may be more suitable at resolving non-linear / sparse features such as those in subsurface density maps. Additionally, this is likely more computationally efficient. To generate a neural net of this kind, one has to train the weights of the neural net via deep learning to perform the inversion. Since there is limited subsurface density maps, one must use synthetic data. The training process requires the follwing:

Generate synthetic subsurface density map → Calculate gravity at surface → Add noise to surface gravity measurement (to simulate realistic survey) → Train neural net:

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

## To-do
#### High Priority
- Refactor ()
- Train & Evaluate (vector vs gz vs gzz vs gz & gzz vs vector & gzz)
- Train & Evaluate (dl vs dl+simpeg vs simpeg)
- train.py
    - loss function
        - MAE
        - dice
        - IoU
        - MSE (regression) & Dice/IoU (segmentation)
        - simpeg forward pass - compare true gravity to predicted gravity
    - data augmentation
        - rotation
        - flips
    - weight decay
    - learning Rate
    - batch size
    - normalization
- StructuralGeo dataset
------
#### Lower Priority
- nn.py
    - dropout
    - skip modules
    - attention
    - 3D --> 3D UNET?
    - multi-sensor fusion
        - ablation study
- generative modelling (Flow-matching & GANs)
- unstructured mesh??? https://www.youtube.com/watch?v=mvKNf_9CYTQ (geotexera)

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

## Open Questions
### Geology
- edge effects?
- synthetic data robustness?
- survey properties (spacing, noise)
- physics limitations (depth, feature size, etc)

### Deep Learning
- alternative architectures?
    - NeRF
    - DeepSDF
    - FNO / DeepONet
    - PINN

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