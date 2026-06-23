# elmod

`elmod` connects the [emcee](https://emcee.readthedocs.io/) MCMC sampler to
[LOC](https://github.com/mjuvela/LOC) for fitting one-dimensional radiative
transfer models to observed spectral lines.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- Astropy
- emcee
- corner
- A working LOC installation and its runtime dependencies

## Installation

Clone the repository and add its parent directory to `PYTHONPATH`:

```sh
git clone https://github.com/ssjensen92/elmod.git
export PYTHONPATH="$PYTHONPATH:$(dirname "$PWD/elmod")"
```

Then import the model class:

```python
from elmod import elmodel

model = elmodel(model_data, ini_file="model.ini", x=velocity, y=spectrum)
```

The model must also be configured with the target beam sizes, LOC output band
names, source velocity, observational errors, and a prior function before an
MCMC run.

## LOC license and citation

LOC is developed by Mika Juvela and is distributed under the GNU General
Public License v3.0. Any use, modification, or redistribution of LOC source
code must comply with the LOC license and preserve its copyright and license
notices. The `elmod` license does not replace or override the license that
applies to LOC.

If you use LOC in scientific work, please cite:

> Juvela, M. (2020), "LOC program for line radiative transfer,"
> *Astronomy & Astrophysics*, **644**, A151.
> https://doi.org/10.1051/0004-6361/202039456

See the [LOC repository](https://github.com/mjuvela/LOC) for its source code,
license, and current documentation.

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).
