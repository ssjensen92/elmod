"""Plot the median spectrum and posterior from the HCN emcee chain.

The spectrum plotting code handles any number of bands. If ``make_model`` in
``hcn_10.py`` is extended with the HCN J=3-2 arrays as described there, this
script automatically adds a second spectrum/residual panel.
"""

import argparse
from pathlib import Path

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from hcn_10 import configure_runtime, make_model


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = [
    r"$\log_{10}(X_{\rm inner})$",
    r"$\log_{10}(X_{\rm outer})$",
]
BAND_LABELS = ["HCN J=1-0", "HCN J=3-2"]


def read_samples(filename, discard=5, thin=1):
    """Read a flattened, finite posterior sample from an emcee backend."""
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    chain = reader.get_chain()
    if discard >= chain.shape[0]:
        raise ValueError(
            f"discard={discard} removes all {chain.shape[0]} saved steps"
        )
    samples = reader.get_chain(discard=discard, thin=thin, flat=True)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if samples.size == 0:
        raise ValueError("the selected chain contains no finite samples")
    return samples


def plot_spectra(model, median, output):
    """Evaluate and plot the median model for every configured band."""
    model_spectrum = np.asarray(model.LOC(median))
    figure, axes = plt.subplots(
        model.nbands,
        1,
        figsize=(9, 4.5 * model.nbands),
        squeeze=False,
    )

    offset = 0
    for band, axis in enumerate(axes[:, 0]):
        size = len(model.V_obs[band])
        predicted = model_spectrum[offset:offset + size]
        observed = np.asarray(model.T_obs[band])
        velocity = np.asarray(model.V_obs[band])
        offset += size

        label = BAND_LABELS[band] if band < len(BAND_LABELS) else f"Band {band}"
        axis.plot(
            velocity,
            observed,
            color="black",
            drawstyle="steps-mid",
            label="Observed", linewidth=2.0
        )
        axis.plot(velocity, predicted, color="tab:red", linewidth=1.5, label="Median model")
        axis.plot(
            velocity,
            observed - predicted,
            color="tab:blue",
            linewidth=2.0,
            label="Residual", drawstyle="steps-mid",
        )
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set_title(label)

        axis.set_xlabel(r"Velocity (km s$^{-1}$)")
        axis.set_ylabel(r"$T_{\rm mb}$ (K)")
        axis.legend()

    if offset != model_spectrum.size:
        raise ValueError("model spectrum length does not match the configured bands")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_corner(samples, median, output):
    """Plot the posterior, using generic labels if parameters are later added."""
    ndim = samples.shape[1]
    labels = DEFAULT_LABELS if ndim == len(DEFAULT_LABELS) else [
        rf"$\theta_{{{index}}}$" for index in range(ndim)
    ]
    figure = corner.corner(samples, labels=labels, truths=median)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chain",
        default=str(EXAMPLE_DIR / "hcn_10.h5"),
        help="emcee HDF backend written by hcn_10.py",
    )
    parser.add_argument(
        "--ini",
        default=str(EXAMPLE_DIR / "HCN.ini"),
        help="LOC configuration used for the fit",
    )
    parser.add_argument("--discard", type=int, default=5)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument(
        "--spectrum-output",
        default=str(EXAMPLE_DIR / "hcn_median.png"),
    )
    parser.add_argument(
        "--corner-output",
        default=str(EXAMPLE_DIR / "hcn_corner.png"),
    )
    args = parser.parse_args()

    if args.thin < 1:
        parser.error("--thin must be at least 1")
    if args.discard < 0:
        parser.error("--discard cannot be negative")
    chain_file = Path(args.chain).resolve()
    spectrum_output = Path(args.spectrum_output).resolve()
    corner_output = Path(args.corner_output).resolve()
    if not chain_file.is_file():
        parser.error("chain not found: " + str(chain_file))

    try:
        ini_file = configure_runtime(args.ini)
        samples = read_samples(chain_file, discard=args.discard, thin=args.thin)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    median = np.median(samples, axis=0)
    model = make_model(ini_file)
    plot_spectra(model, median, spectrum_output)
    plot_corner(samples, median, corner_output)
    print("Median parameters:", median)
    print("Wrote", spectrum_output)
    print("Wrote", corner_output)


if __name__ == "__main__":
    main()
