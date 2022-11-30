"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono


Classes and functions to process multiple HDF5 files from Dewesoft.

Author:
    Fabio Casagrande Hirono
    November 2022
"""


import h5py
import numpy as np

import soundfile as sf
import scipy.signal as ss


from sma_singlefile import SingleFileTimeSeries, SingleFilePSD
from sma_consts_aux import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, \
    DEFAULT_WINDOW, _calc_spectral_centroid, _round_to_nearest_odd



# #############################################################################
# %% Class 'MultiFileTimeSeries'
# #############################################################################

class MultiFileTimeSeries:
    """
    Class to read multiple raw measurement data from Dewesoft HDF5 files
    """

    # *************************************************************************
    def __init__(self, filenames, mic_channel_names, T=30, fs=50000,
                 other_ch_names=None, fs2=None):

        # list of names of files to be read (must be HDF5)
        self.filenames = filenames
        self.N_files = len(self.filenames)

        # list of microphone channels' names in all files
        self.mic_channel_names = mic_channel_names
        self.N_ch = len(self.mic_channel_names)

        # nominal duration of data recording, in seconds
        #   float
        self.T = T

        # default sampling freq
        #   float
        self.fs = fs

        # time vector
        #   (T*fs,) array
        self.t = np.linspace(0, self.T - 1/self.fs, self.T*self.fs)


        # list of non-acoustic channels in 'filename'
        #   list or None
        self.other_ch_names = other_ch_names

        # 2nd sampling freq, for data acquired with SIRIUSiwe STG-M rack unit
        # (e.g. load cell, thermocouple)
        #   float or None
        self.fs2 = fs2

        if fs2:
            #   (T*fs2,) array
            self.t2 = np.linspace(0, self.T - 1/self.fs2, self.T*self.fs2)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # Read files

        # create list of files' data, append list with contents from each file
        self.files = []

        for fi in range(self.N_files):
            file = SingleFileTimeSeries(filenames[fi], self.mic_channel_names,
                                        self.T, self.fs, self.other_ch_names,
                                        self.fs2)
            self.files.append(file)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # if given list of non-acoustic channels, read their mean values
        if other_ch_names:
            self.calc_channel_mean(other_ch_names)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    # *************************************************************************
    def filter_data(self, filter_order=3, fc=50, btype='highpass'):
        """
        Filter time-domain microphone data at given filter order, cutoff
        frequency(ies), filter type, and overwrite result over original data.
        Uses Butterworth filter topology, and applies fwd-bkwd filtering.
        """

        # iterates over list of files, calling 'filter_data' on each one
        for fi in range(self.N_files):
            self.files[fi].filter_data(filter_order, fc, btype)


    # *************************************************************************
    def calc_channel_mean(self, ch_names):
        """
        Iterates over a list of channel names and calculates their mean value
        over time. Generally used for non-acoustic data - e.g. temperature,
        load cells, etc. (Wrapper method for 'calc_channel_mean' of each
        'SingleFileTimeSeries' instance.)
        """

        # # iterates over list of files - already done in SingleFileTimeSeries init
        # for fi in range(self.N_files):
        #     self.files[fi].calc_channel_mean(ch_names)

        for name in ch_names:
            attr_values = np.zeros(self.N_files)
            for fi in range(self.N_files):
                attr_values[fi] = getattr(self.files[fi], 'mean_'+name)

            setattr(self, 'mean_' + name, attr_values)


# #############################################################################
# %% Class 'MultiFilePSDs'
# #############################################################################

class MultiFilePSD:
    """
    Class to store post-processed frequency-domain power spectral density data
    from multiple files, collected over a list of filenames.
    """

    def __init__(self, filenames, mic_channel_names, Ndft=DEFAULT_NDFT,
                 Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):

        self.filenames = filenames
        self.N_files = len(self.filenames)

        self.mic_channel_names = mic_channel_names

        self.Ndft = Ndft
        self.window = window
        self.Noverlap = Noverlap

        self.calc_PSDs(Ndft, window, Noverlap)


    # *************************************************************************
    def calc_PSDs(self, Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP,
                  window=DEFAULT_WINDOW):
        """
        Calculates PSDs for all files in list of file names.
        """

        # create list of 'SingleFilePSD' objects
        self.files = []

        # calculate PSDs from remaining files
        for fi in range(self.N_files):

            ds_data = SingleFileTimeSeries(self.filenames[fi],
                                           self.mic_channel_names)

            self.files.append(ds_data.calc_PSDs(Ndft, window, Noverlap))

        # brings some metadata to current namespace
        self.N_ch = (self.files[0].psd).shape[0]
        self.fs = self.files[0].fs
        self.df = self.files[0].df
        self.freq = self.files[0].freq


    # *************************************************************************
    def calc_broadband_SPL(self, f_low, f_high, kernel_size=100, units='Hz'):
        """
        Calculates the broadband SPL within a frequency band ['f_low', 'f_high']
        by integrating the broadband PSDs within this band. Broadband PSDs are
        calculated using a median filter method, and results are returned
        in dB re 20 uPa RMS.


        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        kernel_size : int, optional
            Size of median filter kernel. The default is 100 Hz.

        units : {'points', 'Hz'}, optional
            Units for kernel size. Default is 'Hz'.

        Returns
        -------
        broadband_SPL : (N_files, N_ch)-shape array_like
            Integrated broadband SPL per file, per channel, in dB re
            20 uPa RMS, within the frequency band [f_low, f_high].
        """

        self.broadband_SPL = np.zeros((self.N_files, self.N_ch))

        for n in range(self.N_files):
            self.files[n].calc_broadband_PSD(kernel_size, units)
            self.broadband_SPL[n, :] = self.files[n].calc_broadband_SPL(f_low, f_high)

        return self.broadband_SPL


    # *************************************************************************
    def calc_overall_SPL(self, f_low, f_high):
        """
        Calculates the overall SPL within a frequency band ['f_low', 'f_high']
        by integrating the PSDs within this band. Results are returned
        in dB re 20 uPa RMS.

        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        Returns
        -------
        overall_SPL : (N_files, N_ch)-shape array_like
            Integrated overall SPL per file, per channel, in dB re 20
            uPa RMS, within the frequency band [f_low, f_high].

        """

        self.overall_SPL = np.zeros((self.N_files, self.N_ch))

        for n in range(self.N_files):
            self.overall_SPL[n, :] = self.files[n].calc_overall_SPL(f_low, f_high)

        return self.overall_SPL


    # *************************************************************************
    def find_peaks(self, f_low, f_high, dB_above_broadband=3):
        """
        Finds peaks in PSD spectrum within a bandwidth [f_low, f_high]. Peaks
        are not restricted to be harmonics of a fundamental frequency. Optional
        arguments are the height above PSD broadband component as threshold.

        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        dB_above_broadband : float, optional
            Minimum peak height above broadband PSD component, in decibels. Default
            value is 3 dB.


        Returns
        -------
        'peak_indices' : (N_files, N_ch, N_peaks)-shape array_like
            Indices of all peaks above threshold.

        'peak_lims' : (N_files, N_ch, N_peaks, 2)-shape array_like
            Lower and upper indices determining the width of each peak.
            Defined as the points where the peak in raw PSD crosses the PSD
            broadband component.
        """

        N_peaks_max = 0

        self.peak_indices = np.zeros((self.N_files, self.N_ch, self.Ndft//2+1),
                                     dtype=int)
        self.peak_lims = np.zeros((self.N_files, self.N_ch, self.Ndft//2+1, 2),
                                  dtype=int)

        for n in range(self.N_files):
            self.files[n].find_peaks(f_low, f_high, dB_above_broadband)

            N_peaks = self.files[n].peak_indices.shape[1]

            self.peak_indices[n, :, :N_peaks] = self.files[n].peak_indices
            self.peak_lims[n, :, :N_peaks, :] = self.files[n].peak_lims

            N_peaks_max = np.max([N_peaks_max, N_peaks])

        # change size of 'peak_indices' to largest no. of peaks found
        temp_peak_indices = np.copy(self.peak_indices[:, :, :N_peaks_max])
        self.peak_indices = np.copy(temp_peak_indices)

        # change size of 'peak_indices' to largest no. of peaks found
        temp_allpeak_lims = np.copy(self.peak_lims[:, :, :N_peaks_max, :])
        self.peak_lims = np.copy(temp_allpeak_lims)

        return self.peak_indices, self.peak_lims


    # *************************************************************************
    def calc_peaks_SPL(self):
        """
        Returns an array of peaks' levels per file, per channel, in
        dB re 20 uPa RMS.

        Parameters
        ----------
        None

        Returns
        -------
        peaks_SPL : (N_files, N_ch, N_peaks)-shape array_like
            Array of integrated tones' SPL per file, per channel, in
            dB re 20 uPa RMS.

        Notes
        -----
        Number of tones can vary per channel and file.
        """

        N_peaks = self.peak_indices.shape[2]

        self.peaks_SPL = np.zeros((self.N_files, self.N_ch, N_peaks))

        for n in range(self.N_files):

            self.peaks_SPL[n, :, :] = self.files[n].calc_peaks_SPL()

        return self.peaks_SPL


    # *************************************************************************
    def calc_tonal_SPL(self):
        """
        Returns the tonal SPL per file, per channel, as the sum of all
        tones (BPF harmonics or other peaks) SPLs.

        Parameters
        ----------
        None

        Returns
        -------
        tonal_SPL : (N_files, N_ch)-shape array_like
            Array of integrated BPF harmonics' SPL per file, per
            channel, in dB re 20 uPa RMS.

        Notes
        -----
        Indices are zero-based, but BPF harmonics are one-based: index 0
        represents 1xBPF, index 1 represents 2xBPF, etc.
        """

        self.tonal_SPL = np.zeros((self.N_files, self.N_ch))

        for n in range(self.N_files):
            self.tonal_SPL[n, :] = self.files[n].calc_tonal_SPL()

        return self.tonal_SPL


    # *************************************************************************
    def SPL_to_polar(self, spl_name, combine_90deg=False):
        """
        Calculates the polar directivity pattern for a given SPL calculation
        by combining measurements made at (azim, elev) = (0, 0-90) and
        (azim, elev) = (180, 0-90).

        Parameters
        ----------
        spl_name : {'overall_SPL', 'broadband_SPL', 'tonal_SPL'}, str
            String containing name of SPL function to output.

        combine_90deg : bool, optional
            Boolean flag denoting whether to combine measurements at 90deg.
            See Notes below.


        Returns
        -------
        polar_angles : (N_angles,)-shape array
            Array containing the elevation angles from 0deg to 180deg. See
            Notes below on how 'combine_90deg' flag affects this array.

        SPL_polar : (N_angles,)-shape array
            Array containing the SPL values over elevation angles. See Notes
            below on how 'combine_90deg' flag affects this array.

        Notes
        -----
        A range of polar angles [0 - 180deg] is created by combining
        measurements made at 0deg azim and 180deg azim, but the two overlapping
        90deg elevation readings are dealt with in one of two ways:

            - if 'combine_90deg = True', they are averaged and shown as a
            single value at 90deg. The resulting arrays will have 'N_angles=19'
            points equally spaced between 0 and 180deg elevation.

            - if 'combine_90deg = False', they are are instead shown separately
            at 89deg and 91deg, in order to assess any differences between
            them. The resulting arrays will have 'N_angles=20' points between
            0 and 180deg elevation, and all except 90deg are equally spaced.
        """

        spl_names_list = ['overall_SPL', 'broadband_SPL', 'tonal_SPL']
        assert spl_name in spl_names_list, "'spl_name' not recognized!"

        SPL_az_elev = getattr(self, spl_name)

        if combine_90deg:
            polar_angles = np.linspace(0, 180, 19)
            SPL_polar = np.zeros((19))

            # write 0deg-80deg (0deg azim in SPL_az_elev)
            SPL_polar[:9] = SPL_az_elev[0, :9]

            # write 100deg-180deg (180deg azim in SPL_az_elev, reversed)
            SPL_polar[10:] = (SPL_az_elev[-1, :9])[::-1]

            # average 90deg elev value from 0deg azim and 180deg azim
            SPL_polar[9] = np.mean([SPL_az_elev[0, 9], SPL_az_elev[-1, 9]])


        else:
            polar_angles_1 = np.linspace(0, 90, 10)
            polar_angles_1[-1] = 89

            polar_angles_2 = np.linspace(90, 180, 10)
            polar_angles_2[0] = 91

            polar_angles = np.concatenate((polar_angles_1, polar_angles_2))

            SPL_polar = np.zeros((20))

            # write 0deg-89deg (0deg azim in SPL_full)
            SPL_polar[:10] = SPL_az_elev[0, :]

            # write 91deg-180deg (180deg azim in SPL_full, reversed)
            SPL_polar[10:] = (SPL_az_elev[-1, :])[::-1]


        return polar_angles, SPL_polar
