## Objective

Resolve geologic density contrast map based on gravity measurements at surface. There are two technques for solving this ill-posed inverse problem.

## Traditional 

Density contrast (mesh, values, topography, survey locations) → Get true gravity → Add noise to simulate realistic gravity measurements → Solve for density contrast using SIMPEG (L2 or IRLS inverse solvers)

[bayesian.ipynb](archive/bayesian.ipynb)

Unfortunately this has many shortcomings and is computationally inneficient.

## Deep Learning

A neural net may be a better and more computationally efficient solution... Although to generate this neural net (to solve for density contrast), requires training with synthetic data:

Density contrast (mesh, values, topography, survey locations) → Get true gravity → Add noise to simulate realistic gravity measurements → Train neural net:

- density contrast (y)
- gravity measurement (x)

### Dataset 

- 20 000 samples
- 3:1:1 training, test, validation
- 0-1g/cm^3
- 32 x 32 x 16 (50m)
- 1-4 blocks
- topography?
- noise?

### Training (UNET 2D --> 3D)

[12000, 1, 32, 32] --> [12000, 1, 32, 32, 16]

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







