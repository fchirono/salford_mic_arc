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

import h5py
import sys

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


def _calc_centroid(x, y):
    """
    Calculates the centroid for two arrays 'x' and 'y', where 'y' is assumed
    to be a function of 'x'.

    This function is meant to be used for roughly estimating tone centre
    frequency when these fall in between frequency samples.
    """
    return np.sum(x*y)/np.sum(y)


def _round_to_nearest_odd(x):
    " Rounds number to nearest odd integer. "
    return int(2*np.floor(x/2)+1)


# ##########################################################################
# %% Functions to explore HDF5 file structure
# ##########################################################################

def print_hdf5_file_structure(file_name):
    """
    Prints the HDF5 file structure
    Functions modified from:
        https://confluence.slac.stanford.edu/display/PSDM/HDF5
    """
    file = h5py.File(file_name, 'r')    # open read-only
    item = file                         # ['/Configure:0000/Run:0000']

    print_hdf5_item_structure(item, offset='')
    file.close()


def print_hdf5_item_structure(g, offset='    '):
    """
    Prints the input file/group/dataset (g) name and begin iterations on its
    content
    Functions modified from:
        https://confluence.slac.stanford.edu/display/PSDM/HDF5
    """
    if isinstance(g, h5py.File):
        print('\n')
        print(g.file)
        print('[File]', g.name)
        print('File attributes:')
        for f_attr in g.attrs.keys():
            print('    ', f_attr, ': ', g.attrs[f_attr])

    elif isinstance(g, h5py.Dataset):
        print('\n')
        print(offset + '[Dataset]', g.name)
        print(offset + 'shape =', g.shape)
        print(offset + 'dtype =', g.dtype.name)
        print(offset + 'Dataset attributes:')
        for d_attr in g.attrs.keys():
            print(offset + '    ', d_attr, ': ', g.attrs[d_attr])

    elif isinstance(g, h5py.Group):
        print('\n')
        print(offset + '[Group]', g.name)
        print(offset + 'Group Members:')
        for g_members in g.keys():
            print(offset + '    ', g_members)

        print(offset + 'Group Attributes:')
        for g_attrs in g.attrs.keys():
            print(offset + '    ', g_attrs)

    else:
        print('WARNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit('EXECUTION IS TERMINATED')

    if isinstance(g, (h5py.File, h5py.Group)):
        for key in g.keys():
            subg = g[key]
            # print(offset, key,)
            print_hdf5_item_structure(subg, offset + '    ')
