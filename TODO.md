# TODO

## Theory

* Consider having sin(theta) instead of theta in differential equations, might be more numerically stable.
* How about surface rotation speed of the planet? Atmosphere moves along, so it is relevant even for airburst.
* Velocity cutoff for ablation depending on air density? ([Passey and Melosh, 1980, p.221](https://www.sciencedirect.com/science/article/pii/001910358090072X))
* According to [Artemieva and Shuvalov (1996)](https://doi.org/10.1007/BF02434011), fragments stay in the near wake of the leader if initial lateral velocity (just after breakup, not V_T from Passey and Melosh (1980), which is after wake separation) U < 0.6*V*sqrt(rho_air/rho_meteoroid), where V is the velocity along the trajectory
* Implement ablation model as in [Crawford (1996)](https://doi.org/10.1017/S0252921100115490)?
* Debris cloud model mentions that when ram pressure drops below initial strength, the tiny fragments form individual bow shocks. -> implement this? How?
* How do we handle the density of porous meteoroids? Do we have to differentiate between average density (including pores) and material density (without the pores) depending on where we use the density?
* `dE/dz` calculation edge case: When a fragment is ascending and getting slower (losing energy due to friction), `dE/dz` is negative! If it's descending and getting slower, `dE/dz` is positive. => Define otherwise? E.g. `dE/dz = (E_2 - E_1) / |z_2 - z_1|` instead of `dE/dz = (E_2 - E_1) / (z_2 - z_1)`?

## Implementation

* Maybe have a look at the [Boost Parameter Library](https://www.boost.org/doc/libs/1_73_0/libs/parameter/doc/html/index.html).
* Debugging a C/C++ extension for python: read this article on [Python Extension Patterns](https://pythonextensionpatterns.readthedocs.io/en/latest/debugging/debug_in_ide.html)
* [Expression templates](https://en.wikipedia.org/wiki/Expression_templates) for lazy evaluation of `offset` operators.
* Use [python development mode](https://docs.python.org/3.9/library/devmode.html) to spot additional warnings
* Look into using odeint [Boost.Numeric.Odeint](https://www.boost.org/doc/libs/1_73_0/libs/numeric/odeint/doc/html/index.html)
* It is possible that close to the ground a fragment splits up into many small pieces (so small that they get removed from the simulation), even though the resulting craters would be combined into one with an observable size. How do we tackle this problem?
* How do we deal with the case where the debris cloud's tangential area is more than half the area of the resulting crater? This happens often enough, can't be ignored.

## Performance

* Do profiling in various use cases first! Consider using this profiler, specifically for python extensions: [yep package](https://github.com/fabianp/yep), which is a wrapper for the [gperftools](https://github.com/gperftools/gperftools) (originally Google Performance Tools) profiler.
* Consider using AVX for `offset * dt`, `offset1 + offset2` and `fragment += offset` [(see this blogpost for example)](https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX).
* More function objects instead of switches in df/dt function.
* Use C++11 concurrency API, task based, for multi-threading? Task = simulate one fragment, add further breakup fragments to a global queue or something?
* For parallelization of the fragments queue, consider Microsoft's [concurrent_queue class](https://docs.microsoft.com/en-us/cpp/parallel/concrt/reference/concurrent-queue-class?view=vs-2019).
