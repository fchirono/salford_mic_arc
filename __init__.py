"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono


Software to read acoustic measurement data performed with a microphone arc
directivity array in the University of Salford anechoic chamber. The mic arc
has a radius of 2.5 metres, and contains 10 microphones located in 10deg
intervals (i.e. from 0deg to 90deg), generally positioned in elevation.

Data is acquired using Dewesoft hardware and software, and the raw time series
are exported as HDF5 files. Data consists of up to 10 channels of acoustic
data, and possibly other channels for auxiliary data (load cells,
thermocouples, etc)


Dependencies:
    - h5py: interface to HDF5 data format
    - numpy: array processing
    - scipy: scientific library
    - soundfile: audio library (for exporting .WAV files)

Author:
    Fabio Casagrande Hirono
    November 2022
"""

import h5py
import numpy as np

import soundfile as sf
import scipy.signal as ss


from .sma_consts_aux import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, DEFAULT_WINDOW, \
    print_hdf5_file_structure, print_hdf5_item_structure

from .sma_singlefile import InputFile, InputFiles, SingleFileTimeSeries,\
    SingleFilePSD
from .sma_multifile import MultiFileTimeSeries, MultiFilePSD

