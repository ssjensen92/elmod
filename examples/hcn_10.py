"""Fit an anonymized starless-core HCN J=1-0 spectrum with elmod.

The physical core model is embedded below, and the observed spectrum is stored
as a two-column ASCII file beside this script. The modified LOC runtime, HCN
configuration, molecular data, and overlap data are bundled with the
repository. Pass ``--ini`` only to use a different LOC configuration.

The default is deliberately a one-band fit.  See ``ADDING HCN J=3-2`` at the
end of this file for the few changes needed for a joint fit.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from elmod import elmodel


AU = 1.495978707e13  # cm
BUNDLED_LOC_DIR = Path(__file__).resolve().parents[1] / "loc"
N_SHELLS = 128
RADIUS_AU = np.linspace(10.0, 60_000.0, N_SHELLS)

# Anonymized physical model.  These density control points replace the
# source-specific .npz file used in the original analysis.
DENSITY_RADIUS_AU = np.array([
    10.0, 3788.9, 7567.8, 11346.7, 15125.6, 18904.5, 22683.4,
    26462.3, 30241.2, 34020.1, 37799.0, 41577.9, 45356.8,
    49135.7, 52914.6, 56693.5, 60000.0,
])
DENSITY_CM3 = np.array([
    51759.781, 47593.254, 37521.078, 26204.539, 16873.172,
    10381.798, 6271.642, 3788.467, 2311.618, 1435.179, 908.832,
    587.552, 388.030, 261.442, 179.595, 125.622, 93.186,
])


def physical_profiles(radius_au):
    """Return fixed radial velocity, density, temperature, and linewidth."""
    density = 10.0 ** np.interp(
        np.log10(radius_au),
        np.log10(DENSITY_RADIUS_AU),
        np.log10(DENSITY_CM3),
    )
    temperature = 8.2837 + (14.6849 - 8.2837) / 2.0 * (
        1.0 + np.tanh((radius_au - 36_441.0) / 33_501.0)
    )
    velocity = 0.6258 * np.exp(
        -0.5 * ((radius_au - 42_943.5) / 17_911.6) ** 2
    )
    linewidth = 0.075964 + (0.105529 - 0.075964) / np.pi * (
        np.pi / 2.0 + np.tanh((radius_au - 49_959.7) / 371.5)
    )
    return velocity, density, temperature, linewidth


VELOCITY, DENSITY, TEMPERATURE, LINEWIDTH = physical_profiles(RADIUS_AU)
MODEL_DATA = np.column_stack((RADIUS_AU * AU, VELOCITY, DENSITY, TEMPERATURE))

# Native-resolution HCN J=1-0 observation. The ASCII velocity axis has the
# original 6.05 km/s systemic velocity removed, matching the source analysis.
HCN_10_SOURCE_VELOCITY = 6.05
HCN_10_MODEL_VELOCITY_OFFSET = 4.843549499074382
HCN_10_RMS = 0.017968176238194116
V_10, T_10 = np.loadtxt(
    Path(__file__).with_name("hcn_10_spectrum.txt"), unpack=True
)


class HCNModel(elmodel):
    """elmodel with a fixed core structure and two abundance parameters."""

    def write_model(self, theta, N=N_SHELLS, model_fname="model.cloud"):
        log_x_inner, log_x_outer = theta
        abundance = np.full(N_SHELLS, 10.0 ** log_x_inner)
        abundance[RADIUS_AU > 1600.0] = 10.0 ** log_x_outer

        cloud = np.column_stack((
            DENSITY,
            TEMPERATURE,
            LINEWIDTH,
            abundance,
            -VELOCITY,
        )).astype(np.float32)
        with open(model_fname, "wb") as handle:
            np.asarray([N_SHELLS], np.int32).tofile(handle)
            np.asarray([RADIUS_AU / RADIUS_AU.max()], np.float32).tofile(handle)
            cloud.tofile(handle)


def log_prior(theta):
    log_x_inner, log_x_outer = theta
    if -9.0 < log_x_inner < -6.0 and -10.5 < log_x_outer < -7.5:
        return 0.0
    return -np.inf


def make_model(ini_file):
    model = HCNModel(MODEL_DATA, ini_file=ini_file, nbands=1)
    model.V_obs[0] = V_10
    model.T_obs[0] = T_10
    model.x = V_10.copy()
    model.y = T_10.copy()
    model.yerr = np.full_like(T_10, 2.0 * HCN_10_RMS)
    model.targ_beams = [27.8]  # arcsec
    model.band_fnames = ["HCN.band0.spe"]
    # Convert LOC's band-0 reference velocity to the same systemic frame as
    # the observation. This offset reproduces the original frequency mapping.
    model.V_lsr = HCN_10_MODEL_VELOCITY_OFFSET
    model.pass_priorfunc(log_prior)
    return model


def configure_runtime(ini_file):
    """Configure the bundled LOC runtime and return an absolute ini path."""
    ini_file = os.path.abspath(ini_file)
    if not os.path.isfile(ini_file):
        raise FileNotFoundError("LOC configuration not found: " + ini_file)

    # LOC resolves molecular/HFS/overlap inputs relative to the configuration,
    # while its Python and OpenCL sources come from elmod's bundled runtime.
    os.chdir(os.path.dirname(ini_file))
    sys.path.insert(0, str(BUNDLED_LOC_DIR))
    os.environ["ELMOD_LOC_DIR"] = str(BUNDLED_LOC_DIR)
    return ini_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ini",
        default=str(Path(__file__).with_name("HCN.ini")),
        help="LOC configuration (default: examples/HCN.ini)",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--burnin", type=int, default=6)
    parser.add_argument("--output", default="hcn_10.h5")
    args = parser.parse_args()

    try:
        ini_file = configure_runtime(args.ini)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    model = make_model(ini_file)
    rng = np.random.default_rng(7)
    start = np.array([-7.25, -8.65])
    walkers = start + rng.normal(0.0, 0.03, size=(8, start.size))
    model.run_mcmc(
        walkers,
        nsteps=args.steps,
        burnin=args.burnin,
        fname=args.output,
    )


if __name__ == "__main__":
    main()


# ADDING HCN J=3-2
# ------------------
# Load or embed V_32 and T_32, then change make_model as follows:
#
#   model = HCNModel(MODEL_DATA, ini_file=ini_file, nbands=2)
#   model.V_obs = [V_10, V_32]
#   model.T_obs = [T_10, T_32]
#   model.x = np.concatenate(model.V_obs)
#   model.y = np.concatenate(model.T_obs)
#   model.yerr = np.concatenate((np.full_like(T_10, 0.05), sigma_32))
#   model.targ_beams = [27.8, 9.3]
#   model.band_fnames = ["HCN.band0.spe", "HCN.band2.spe"]
#
# The LOC ini file must write both bands; in the configuration from which this
# example was derived, band0 is HCN J=1-0 and band2 is HCN J=3-2.  elmod then
# evaluates both spectra in one likelihood, so the physical and abundance
# parameters are constrained jointly.
