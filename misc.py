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
