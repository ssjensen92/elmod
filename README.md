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

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).
