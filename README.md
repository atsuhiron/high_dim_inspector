# high_dim_inspector
This is a set of tools for inspecting distribution of data points in higher dimensions.

## 1. Requirements
Python >= 3.10

## 2. Common rules
Data is represented by `numpy.ndarray`.
The shape of the dataset is `(N, D)` where `N` is the number of data and `D` is the number of dimensions.

## 3. Description
### 3-1. gen_data
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

### 3-2. sparse_dense_booster
This script is used to amplify the sparseness and density of the distribution of data points.

```python
import numpy as np
import sparse_dense_booster as sdb

points = np.random.random((10, 2)).astype(np.float32)

param = sdb.create_sdb_param(
    points,
    base_amp=0.6,    ## Coefficient of force strength
    t_end=0.2        ## Time to act on amplification
)
boosted_points = sdb.boost(points, param)
```

`boosted_points` are data points whose sparsity and density have been amplified.
The amplification action is quite sensitive to the value of `SDBParam`, so you will most likely have to repeat the trial-and-error process.
Normally, helper functions like the above are used, but it is also possible to set all values manually.

```python
import sparse_dense_booster as sdb

param = sdb.SDBParam(
    iter_num=10,      ## Number of iteration
    min_dist=0.05,    ## Distance at which the gravitational force reaches its maximum value
    pot_peak=0.6,     ## Distance where attraction and repulsion switch
    amplitude=0.6,    ## Coefficient of force strength
    delta_t=0.01      ## time step size
)
```

#### 3-2-1. Mathematics
The data is considered to be a collection of point mass with the same mass and is set in motion by a virtual force $\vec{f}$.
The coordinates $\vec{x}^{(t + 1)}_{i}$ of mass $i$ at time $t+1$ are expressed by the following equation:

```math
\vec{x}^{(t + 1)}_{i} = \frac{1}{2} \sum_{j \neq i} \vec{f}^{(t)}_{ij} \Delta t^2 + \vec{x}^{(0)}_i
```

where $\Delta t$ and $\vec{x}^{(0)}_i$ are the time step size and the initial position of the point mass $i$, respectively.

For simplicity, subscripts indicating time $(t)$ are omitted below. 
The force $f$ acting varies with the distance to each other point mass.
We express the relative position vector of two point masses $i, j$ as $\vec{r} _{ij} = \vec{x} _{j} - \vec{x} _{i}$ and its $L^2$ norm as $r _{ij} = |\vec{r} _{ij}|$,
the force is expressed as follows:

```math
\vec{f}_{ij} = A \left( \frac{1}{r^2_{ij}} - \frac{1}{R^2_p}\right) \frac{\vec{r}_{ij}}{r_{ij}}
```

where $A$ and $R_p$ are constants. Let $A$ be a proportionality constant multiplied by the force, 
and let $d$ be the dimensionality of the data, expressed as:

```math
A = \sqrt{\frac{d}{2}} {\rm (base\_amp)}
```
.
The value of `pot_peak` is used as is for $R_p$ directly.  
When $R_{ij}$ exceeds the value of $R_p$ (i.e., $i$ and $j$ are located farther apart),
the sign of $\vec{f}$ reverses and the attraction changes to repulsion.
This causes nearby pairs to move closer each other and distant pairs to move further apart.

In addition, the coordinates of point mass calculated by the first equation are not used as they are, but are normalized by the following equation: 

```math
\hat{\vec{x}}_{(d)} = (\vec{x}_{(d)} - \vec{\mu}_{(d)}) \oslash \vec{\sigma}_{(d)}  
```

where $\vec{\mu}$ and $\vec{\sigma}$ are vectors representing the mean and standard deviation, respectively, 
the subscript $(d)$ means that the value is in each dimension 
(That is, in numpy notation, $\vec{x}_ {i}$ is represented by `points[i, :]` and $\vec{x} _{(d)}$ by `points[:, d]`).
$\oslash$ means _Hadamard_ division.  
Skipping this normalization step will result in an infinite diffusion of the pawn points, depending on the value of `pot_peak`.