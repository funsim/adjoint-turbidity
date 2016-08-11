AdjTurbidity 1.0
================

This repository contains the model AdjTurbidity 1.0, described in the paper

*Samuel D. Parkinson, Simon W. Funke, Jon Hill, Matthew D. Piggott, and Peter A. Allison*, **Application of the adjoint approach to optimise the initial conditions of a turbidity current**, Geoscientific Model Development (GMD).


Installation instructions
-------------------------

    1. Install FEniCS 1.3 from http://fenicsproject.org/
    2. Install libadjoint from https://bitbucket.org/dolfin-adjoint/libadjoint. Check out commit 3a99e45533d1ecb3a828c8072d22067d0a66d5b0.
    3. Install dolfin-adjoint from https://bitbucket.org/dolfin-adjoint/dolfin-adjoint. Check out commit d9d3d28632c56e9a68b48f73e2a622a5077b918c.
    4. Install the most recent version of libspud from the Fluidity project https://github.com/FluidityProject/fluidity
    5. Install pyipopt from https://github.com/xuy/pyipopt.
    5. Clone this repository and include its root path to $PYTHONPATH


Simulation scripts
------------------

* Section 2.5 Forward model verification
    Reproduce with:

```
#!bash
    $ cd tests/similarity
    $ python similarity.py
    $ python plot.py
```


* Section 4.3 Verification of the gradient calculation

```
#!bash
    Reproduce with:
    $ cd tests/taylor
    $ python taylor.py
```

* Section 4.4 and 4.6: Optimisation of a model with one/two sediment class(es)
    Reproduce with:

```
#!bash
    $ cd jobs/bed_11
    $ python bed_4_test.py "1"
    $ ./MAKEFILE
    $ ./MAKEFILE2
```