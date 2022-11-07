"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono

Auxiliary file to hold global constants

Author:
    Fabio Casagrande Hirono
    Nov 2022
"""

# Reference acoustic pressure, in Pa RMS
P_REF = 20e-6

# Default DFT size
DEFAULT_NDFT = 2**13

# Default overlap length (for PSDs)
DEFAULT_NOVERLAP = 2**12

# default window
DEFAULT_WINDOW = 'hann'

