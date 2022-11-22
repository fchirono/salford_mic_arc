"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono

Auxiliary file to hold global constants and auxiliary functions

Author:
    Fabio Casagrande Hirono
    Nov 2022
"""

import numpy as np

# ##########################################################################
# %% Global constants
# ##########################################################################

# Reference acoustic pressure, in Pa RMS
P_REF = 20e-6

# Default DFT size
DEFAULT_NDFT = 2**13

# Default overlap length (for PSDs)
DEFAULT_NOVERLAP = 2**12

# default window
DEFAULT_WINDOW = 'hann'


# ##########################################################################
# %% Auxiliary functions
# ##########################################################################


def _calc_spectral_centroid(x, y):
    """
    Calculates the spectral centroid for two arrays 'x' and 'y', where 'y'
    is assumed to be a function of 'x'.

    This function is meant to be used for roughly estimating tone centre
    frequency when these fall in between frequency samples.
    """
    return np.sum(x*y)/np.sum(y)


def _round_to_nearest_odd(x):
    " Rounds number to nearest odd integer. "
    return int(2*np.floor(x/2)+1)
