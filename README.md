# high_dim_inspector
This is a set of tools for inspecting distribution of data points in higher dimensions.

## Common rules
Data is represented by `numpy.ndarray`.
The shape of the dataset is `(N, D)` where `N` is the number of data and `D` is the number of dimensions.

## Description
### gen_data
Script to generate sample data or data for evaluation of other tools.

```python
import gen_data

param = gen_data.GenDataParam(
    num=3000,               ## Number of data points to be generated
    size=(32, 32),          ## Resolution of the data point distribution function used to generate the data
    power=0.6,              ## Exponent to multiply the noise at each scale used to generate the data point distribution function
    max_scale=12,           ## Maximum size of the Gaussian filter used to generate the data point distribution function
    attenuation_coef=0.001, ## Limb attenuation intensity
    use_float32=True        ## np.float32 use or not use
)

sampling_points, distribution = gen_data.gen_data_points(param)
```

`sampling_points` is an array of data points, whose shape is `(num, len(size))`.

`distribution` is the data point distribution function and has the same shape as `size`.
> [!NOTE]
> The `distribution` represents the probability that a given a data point will be produced at the grid,
> not the **probability distribution function** used in the context of statistics.(Integrating over the whole space does not equal `1`.)

### sparse_dense_booster
This script is used to amplify the sparseness and density of the distribution of data points.

```python
import numpy as np
import sparse_dense_booster as sdb

param = sdb.SDBParam(
        iter_num=10,      ## Number of iteration
        min_dist=0.05,    ## Distance at which the gravitational force reaches its maximum value
        pot_peak=0.6,     ## Distance where attraction and repulsion switch
        amplitude=0.6,    ## Coefficient of force strength
        delta_t=0.01      ## time step size
    )

points = np.random.random((10, 2)).astype(np.float32)
boosted_points = sdb.boost(points, param)
```

`boosted_points` are data points whose sparsity and density have been amplified.
The amplification action is quite sensitive to the value of `SDBParam`, so you will most likely have to repeat the trial-and-error process.