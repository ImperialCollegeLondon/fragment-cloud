# Fragment-Cloud Model

Implementation of the fragment-cloud model (FCM) for meteoroid atmospheric entry modeling.
The theoretical model is described in detail in Wheeler et al. (2018) [DOI:10.1016/j.icarus.2018.06.014](https://doi.org/10.1016/j.icarus.2018.06.014)

We added the capability to compute the location and size of impact craters.

## Installation

### Prerequisites

You need to have the following packages installed:

* `python3` v. 3.7 or later with `numpy` and the python development headers
* `C++17` compiler, e.g. `g++` or `clang`
* `cmake` v. 3.12 or later
* `boost` v. 1.70 or later, including the `python3`, `numpy`, and `unit_test_framework` components.

### Instructions

1. Open command line and navigate to the `fragment-cloud/` folder.
2. Inside the `fragment-cloud/` folder, create a new folder called `debug/` or `release/` and `cd` into it.
3. Run the following command: `cmake ..` This will check all dependencies and create a file called `setup.py` inside the `fragment-cloud/` folder.
4. Navigate to the `fragment-cloud/` folder.

For the final step, there are a few different options:

* Install the `fcm` software like a pip package: `python3 setup.py install`. Then the `fcm` package will be available in your default python path.
* Only compile the C++ component: `python3 setup.py build_ext --inplace`. Then you have to add the `fragment-cloud` folder to `sys.path` in order to import the `fcm` module in python, or run scripts from within the `fragment-cloud` folder.
* Build the `fcm` package in a separate build folder: `python3 setup.py build`. This will create a `build/` folder and place all package files in there.

## Tests

1. Navigate to an empty folder called `debug/` or `release/` inside the `fragment-cloud/` folder.
2. Run `cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=1 ..` This will instruct the compiler to use debug flags, which makes it possible to run them with a debugger.
3. `cmake --build .`
4. `ctest`

## Documentation

Sorry, no time for a separate documentation :( The docstrings are comprehensive though. Together with examples it should be clear what is going on. Docstrings are in `fcm/_fcm_class.py` and `fcm/models.py`. 

## Example usage

```python3
import fcm
import fcm.atmosphere as atm

# Load atmospheric density vs. elevation data
atmosphere = atm.US_standard_atmosphere()

# Default parameters
parameters = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=atmosphere)

# Define a meteoroid
impactor = fcm.FragmentationMeteoroid(velocity=15, angle=45, density=3000, radius=1, strength=1e4)

# Simulate impact
results = fcm.simulate_impact(parameters, impactor, h_start=100)

# Visualise results
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 1. dE/dz
plt.plot(results.energy_deposition.to_numpy(), results.energy_deposition.index)
plt.xscale('log')
plt.xlabel('dE/dz (kt TNT/km)')
plt.xlabel('z (km)')
plt.show()

# 2. Crater cluster
results.craters.plot.scatter('x', 'y')
fig, ax = plt.subplots(1, 1)
for row in craters.itertuples():
    circle = Circle((row.y, row.x), row.r, color='black', fill=False)
    ax.add_artist(circle)
```
