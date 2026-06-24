"""
Constants for use in the EMCEE_LOC.py script
=============================

"""

from astropy import constants as con
import os, sys
import numpy as np
from contextlib import contextmanager

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

pc_to_au = con.pc.to('au').value
au_to_cm = con.au.to('cm').value


#########################################################################################################################
# LOC aux. FUNCTIONs, you likely know these
def convolve_loc_sps(filename, fwhm_as, angle_as=None, samples=None):
    '''
        Convolve the spectra with a given FWHM and angle
        Usage:
            V, T = convolve_loc_sps(filename, fwhm_as, angle_as=None, samples=None)
        Input:
            filename = name of the spectrum file
            fwhm_as  = FWHM in arcsec
            angle_as = angle in arcsec of the model
            samples  = number of samples
        Return:
            V  = vector of velocity values, one per channel
            T  = spectra as a cube T[NRAY, NCHN] for NRAY lines of sight and NCHN spectral channels
    '''
    loc_dir = os.environ.get('ELMOD_LOC_DIR')
    if loc_dir:
        loc_dir = os.path.abspath(os.path.expanduser(loc_dir))
        if loc_dir not in sys.path:
            sys.path.insert(0, loc_dir)
    sys.path.append(r'')
    from LOC_aux import ConvolveSpectra1D
    fwhm_as = float(fwhm_as)
    if angle_as is None:
        angle_as = -1
    else:
        angle_as = float(angle_as)
    if samples is None:
        samples = 201
    else:
        samples = int(samples)
    with suppress_output():
        ConvolveSpectra1D(filename, fwhm_as, GPU=0, platforms=[
                      0, 1, 2, 3, 4], angle_as=angle_as, samples=samples)
        V, T = LOC_read_spectra_1D(filename[:-4]+'_convolved.spe')
    return V, T
    
def LOC_read_spectra_1D(filename):
    """
    Read spectra written by LOC1D.py (spherical models).
    Usage:
        V, S = LOC_read_spectra_1D(filename)
    Input:
        filename = name of the spectrum file
    Return:
        V  = vector of velocity values, one per channel
        S  = spectra as a cube S[NRAY, NCHN] for NRAY lines of sight and
             NCHN spectral channels
    """
    fp = open(filename, 'rb')
    NRAY, NCHN = np.fromfile(fp, np.int32, 2)
    V0, DV = np.fromfile(fp, np.float32, 2)
    SPE = np.fromfile(fp, np.float32, NRAY*NCHN).reshape(NRAY, NCHN)
    fp.close()
    return V0+np.arange(NCHN)*DV, SPE


def channel_average_spectrum(velocity, spectrum, centers, channel_width):
    """Average a model spectrum over finite-width observational channels.

    The model is treated as piecewise linear between its native velocity
    samples and integrated exactly over a top-hat channel response. Values
    beyond the model grid use the nearest edge value, matching elmod's former
    interpolation behaviour.
    """
    velocity = np.asarray(velocity, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.broadcast_to(np.asarray(channel_width, dtype=float), centers.shape)

    if velocity.ndim != 1 or spectrum.shape != velocity.shape:
        raise ValueError('velocity and spectrum must be one-dimensional arrays of equal length')
    if velocity.size < 2:
        raise ValueError('at least two model velocity samples are required')
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError('channel widths must be finite and positive')

    order = np.argsort(velocity)
    x = velocity[order]
    y = spectrum[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError('model velocities must be unique')

    dx = np.diff(x)
    slope = np.diff(y) / dx
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (y[:-1] + y[1:]) * dx),
    ))

    def antiderivative(points):
        points = np.asarray(points, dtype=float)
        result = np.empty_like(points)
        left = points <= x[0]
        right = points >= x[-1]
        middle = ~(left | right)
        result[left] = (points[left] - x[0]) * y[0]
        result[right] = cumulative[-1] + (points[right] - x[-1]) * y[-1]
        indices = np.searchsorted(x, points[middle], side='right') - 1
        local_dx = points[middle] - x[indices]
        result[middle] = (
            cumulative[indices]
            + y[indices] * local_dx
            + 0.5 * slope[indices] * local_dx**2
        )
        return result

    lower = centers - 0.5 * widths
    upper = centers + 0.5 * widths
    return (antiderivative(upper) - antiderivative(lower)) / widths
