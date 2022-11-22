# -*- coding: utf-8 -*-
"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono

Class and functions to handle Power Spectral Density values and calculations.

Author:
    Fabio Casagrande Hirono
    Nov 2022
"""

import h5py
import numpy as np
import scipy.signal as ss


import salford_mic_arc as SMA
from sal_constants import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, DEFAULT_WINDOW


# #############################################################################
# %% Class 'SingleFilePSD'
# #############################################################################

class SingleFilePSD:
    """
    Class to store single-file, multichannel PSD and associated frequency-domain
    information. PSDs are assumed single-sided.
    """

    def __init__(self, psd, freq, fs, Ndft=DEFAULT_NDFT,
                 Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):

        # Array of Power Spectral Density values (single sided)
        #   (N_mics, Ndft//2+1)-shape array_like
        self.psd = np.atleast_2d(psd)

        # number of microphone channels
        self.N_ch = self.psd.shape[0]

        # frequency vector (single-sided)
        #   (Ndft//2+1,)-shape array_like
        self.freq = freq

        # sampling frequency
        #   int
        self.fs = fs

        # DFT size
        #   int
        self.Ndft = Ndft

        # Overlap size
        #   int
        self.Noverlap = Noverlap

        # window function (name or samples)
        # str, (Ndft,)-shape array_like
        self.window = window

        # frequency resolution
        #   float
        self.df = self.fs/self.Ndft


    # *************************************************************************
    def calc_broadband_PSD(self, kernel_size=100, units='Hz'):
        """
        Calculates broadband components of multichannel PSD using median
        filtering. This technique removes the contribution of tonal peaks.

        Parameters
        ----------
        kernel_size : int, optional
            Size of median filter kernel. The default is 100 Hz.

        units : {'points', 'Hz'}
            Units for kernel size. Default is 'Hz'.

        Returns
        -------
        None.

        """

        assert units in ['points', 'Hz'], \
            "Unknown input for 'units' - must be 'points' or 'Hz' !"

        # if kernel size is given in Hz, calculate equivalent length in points
        if units == 'Hz':
            kernel_size_Hz = np.copy(kernel_size)
            kernel_size = round_to_nearest_odd(kernel_size_Hz/self.df)

        self.psd_broadband = np.zeros((self.N_ch, self.Ndft//2+1))

        for ch in range(self.N_ch):
            self.psd_broadband[ch, :] = ss.medfilt(self.psd[ch, :], kernel_size)


    # *************************************************************************
    def find_peaks(self, f_low, f_high, dB_above_broadband=3):
        """
        Find peaks in PSD spectrum within a bandwidth [f_low, f_high]. Peaks
        are not restricted to be harmonics of a fundamental frequenycy.
        Optional arguments are the height above PSD broadband component as
        threshold.

        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        dB_above_broadband : float, optional
            Minimum peak height above broadband PSD component, in decibels.
            Default value is 3 dB.


        Returns
        -------
        peak_indices : (N_ch, N_peaks)-shape array_like
            Array of indices for all peaks above threshold.

        peak_lims : (N_ch, N_peaks, 2)-shape array_like
            Lower and upper indices determining the width of each peak.
            Defined as the points where the peak in raw PSD crosses the PSD
            broadband component.
        """

        # assert instance has psd broadband defined
        assert hasattr(self, 'psd_broadband'), \
            "Cannot find peaks: PSD instance does not have attribute 'psd_broadband'!"

        gain_above_broadband = 10**(dB_above_broadband/10)

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        # initialize 'peak_indices' with a large size (Ndft/2+1), reduce it later
        self.peak_indices = np.zeros((self.N_ch, self.Ndft//2+1), dtype=int)

        N_peaks = 0

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        for ch in range(self.N_ch):
            height_range = (self.psd_broadband[ch, freq_mask]*gain_above_broadband,
                            None)

            peak_indices_ch, peak_properties = ss.find_peaks(self.psd[ch, freq_mask],
                                                             height=height_range)

            # number of peaks found in this ch
            N_peaks_ch = peak_indices_ch.shape[0]
            self.peak_indices[ch, :N_peaks_ch] = peak_indices_ch

            # largest number of peaks found so far
            N_peaks = np.max([N_peaks, N_peaks_ch])

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # change size of 'peak_indices' to largest no. of peaks found
        temp_peaks = np.copy(self.peak_indices[:, :N_peaks])
        self.peak_indices = np.copy(temp_peaks)

        # add initial index of freq_mask to all non-zero entries
        self.peak_indices[self.peak_indices!=0] += np.argwhere(freq_mask)[0, 0]

        # replace zeros with '-1' as flag for 'no peak found'
        self.peak_indices[self.peak_indices==0] = -1

        # find peak limits
        self.peak_lims = self._find_peak_lims(self.peak_indices)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

        return self.peak_indices, self.peak_lims


    # *************************************************************************
    def _find_peak_lims(self, peak_indices, radius=20, units='points'):
        """
        For a list of peaks in 'psd', given by 'peak_indices', finds a list
        of lower and upper indices to determine peak widths.

        Parameters
        ----------
        peak_indices : (N_ch, N_peaks,)-shape array_like
            Array containing the peak indices per channel.

        radius : int or float, optional
            Search radius for peak limits. Default is 20 points.

        units : {'points', 'Hz'}, optional
            Units for search radius. Default is 'points'.


        Returns
        -------
        peak_lims : (N_ch, N_peaks, 2)-shaped array_like
            Array containing the indices for lower and upper limits of each
            peak, per channel.
        """

        assert units in ['points', 'Hz'], \
            "Unknown input for 'units' - must be 'points' or 'Hz' !"

        # if kernel size is given in Hz, calculate equivalent length in points
        if units == 'Hz':
            radius_Hz = np.copy(radius)
            radius = round_to_nearest_odd(radius_Hz/self.df)

        N_data = (self.psd).shape[1]

        N_peaks = peak_indices.shape[1]

        peak_lims = np.zeros((self.N_ch, N_peaks, 2), dtype=np.int64)

        for ch in range(self.N_ch):
            for n_pk, peak_index in enumerate(peak_indices[ch, :]):

                # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
                # if peak index is '-1', write '-1' on peak lims too
                if peak_index == -1:
                    peak_lims[ch, n_pk, :] = -1

                else:
                    # .........................................................
                    # If peak is closer than 'f_radius' to index 0, use index 0
                    # as peak lim
                    if (peak_index - radius)<=0:
                        peak_lims[ch, n_pk, 0] = 0

                    else:
                        # Region *below* 'peak_index' where 'psd' is lower or equal
                        # to 'psd_broadband'
                        cond_lower = (self.psd[ch, peak_index - radius : peak_index]
                                      <= self.psd_broadband[ch, peak_index - radius : peak_index]).nonzero()[0]


                        # if no value found, take lower edge of search radius
                        if cond_lower.size == 0:
                            lower_lim = -radius

                        # If one or more values found, take last element as lower edge of peak
                        else:
                            lower_lim = cond_lower[-1]

                        peak_lims[ch, n_pk, 0] = lower_lim + (peak_index - radius)

                    # .........................................................
                    # If peak is closer than 'f_radius' to 'N_data', use 'N_data'
                    # as peak lim
                    if (peak_index + radius+1) >= N_data:
                        peak_lims[ch, n_pk, 1] = N_data

                    else:
                        # Region *above* 'peak_index' where 'psd' is lower or equal to
                        # 'psd_broadband'
                        cond_upper = (self.psd[ch, peak_index : peak_index + radius + 1]
                                      <= self.psd_broadband[ch, peak_index : peak_index + radius + 1]).nonzero()[0]

                        # if no value found, take upper edge of search radius
                        if cond_upper.size == 0:
                            upper_lim = + radius

                        # If one or more values found, take first element as upper edge of peak
                        else:
                            upper_lim = cond_upper[0]

                        peak_lims[ch, n_pk, 1] = upper_lim + peak_index

                # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
                # check if peak_lims are identical to any previous peak,
                # replace with -1 if so
                if ( peak_lims[ch, n_pk, :] in peak_lims[ch, :n_pk, :]):
                    peak_lims[ch, n_pk, :] = -1
                # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

        return peak_lims


    # *************************************************************************
    def calc_broadband_SPL(self, f_low, f_high):
        """
        Returns array of integrated broadband SPL per channel, in dB re 20 uPa
        RMS, within a frequency range [f_low, f_high].

        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        Returns
        -------
        broadband_SPL : (N_ch,)-shape array_like
            Integrated broadband SPL per channel, in dB re 20 uPa RMS, within
            the frequency band [f_low, f_high].
        """

        # assert instance has psd broadband defined
        assert hasattr(self, 'psd_broadband'), \
            "Cannot calculate broadband SPL: MutiChannelPSD instance does not have attribute 'psd_broadband'!"

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        integrated_broadband_psd = np.sum(self.psd_broadband[:, freq_mask],
                                          axis=1)*self.df

        self.broadband_SPL = 10*np.log10(integrated_broadband_psd/(P_REF**2))

        return self.broadband_SPL


    # *************************************************************************
    def calc_overall_SPL(self, f_low, f_high):
        """
        Returns integrated overall SPL per channel, in dB re 20 uPa RMS, within
        a frequency range [f_low, f_high].

        Parameters
        ----------
        f_low : float
            Low frequency limit, in Hz.

        f_high : float
            High frequency limit, in Hz.

        Returns
        -------
        overall_SPL : (N_ch,)-shape array_like
            Integrated overall SPL per channel, in dB re 20 uPa RMS, within
            the frequency band [f_low, f_high].
        """

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        integrated_oa_psd = np.sum(self.psd[:, freq_mask], axis=1)*self.df

        self.overall_SPL = 10*np.log10(integrated_oa_psd/(P_REF**2))

        return self.overall_SPL


    # *************************************************************************
    def _calc_peaks_SPL(self):
        """
        Returns array of all peaks' SPL per channel, in dB re 20 uPa RMS.

        Parameters
        ----------
        None

        Returns
        -------
        peaks_SPL : (N_ch, N_peaks)-shape array_like
            Array of integrated peaks' SPL per channel, in dB re 20 uPa RMS.

        Notes
        -----
        Number of tones can vary across channels. If a given peak is not present
        on a channel, its SPL is NaN.
        """

        assert hasattr(self, 'peak_lims'), \
            "Cannot calculate peaks' SPL: SingleFilePSD instance does not have attribute 'peak_lims'!"

        N_peaks = self.peak_indices.shape[1]

        self.peaks_SPL = np.zeros((self.N_ch, N_peaks))

        for ch in range(self.N_ch):
            for n_pk in range(N_peaks):

                # if peak lims is -1 (no peak found), SPL is NaN
                if self.peak_lims[ch, n_pk, 0] == -1:
                    self.peaks_SPL[ch, n_pk] = np.nan

                else:
                    peak_range = np.arange(self.peak_lims[ch, n_pk, 0],
                                           self.peak_lims[ch, n_pk, 1]+1)

                    # subtract broadband content from PSD peak
                    peak_minus_bb = (self.psd[ch, peak_range]
                                     - self.psd_broadband[ch, peak_range])

                    integrated_peak_psd = np.sum(peak_minus_bb)*self.df

                    self.peaks_SPL[ch, n_pk] = 10*np.log10(integrated_peak_psd/(P_REF**2))

        return self.peaks_SPL


    # *************************************************************************
    def calc_tonal_SPL(self):
        """
        Returns the tonal SPL per channel, as the sum of all peaks' SPLs.

        Parameters
        ----------
        None

        Returns
        -------
        tonal_SPL : (N_ch,)-shape array_like
            Array of integrated peaks' SPL per channel, in dB re 20 uPa.
            RMS.

        Notes
        -----
        Must be called after 'find_peaks' method.
        """

        assert hasattr(self, 'peak_lims'),\
            "Cannot calculate tonal SPL: SingleFilePSD instance does not have attribute 'peak_lims'!"

        self.tonal_SPL = np.zeros(self.N_ch)

        self._calc_peaks_SPL()
        peaks_SPL = self.peaks_SPL

        for ch in range(self.N_ch):

            # sum of tones' squared pressures (ignoring NaNs)
            nan_mask = np.isnan(peaks_SPL[ch, :])
            peaks_valid = peaks_SPL[ch, :][~nan_mask]

            sum_bpfs = np.sum(10**(peaks_valid/10))*(P_REF**2)
            self.tonal_SPL[ch] = 10*np.log10(sum_bpfs/(P_REF**2))

        return self.tonal_SPL


# #############################################################################
# %% Class 'MultiFilePSDs'
# #############################################################################

class MultiFilePSDs:
    """
    Class to store post-processed frequency-domain power spectral density data
    from multiple files, collected over a list of azimuthal angles.
    """

    def __init__(self, filenames, azim_angles, mic_channel_names,
                 Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP,
                 window=DEFAULT_WINDOW):

        # TODO: incorporate mic channel names! (separate multi-file reader class?)

        self.filenames = filenames

        self.azim_angles = azim_angles
        self.N_azim = len(azim_angles)

        assert len(self.filenames) == self.N_azim, \
            "Number of elements in 'filenames' and 'azim_angles' does not match!"

        self.Ndft = Ndft
        self.window = window
        self.Noverlap = Noverlap

        self.calc_azim_PSDs(Ndft, window, Noverlap)


    def calc_azim_PSDs(self, Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP,
                        window=DEFAULT_WINDOW):
        """
        Calculates PSDs for all files in list of file names.
        """

        # create list of 'SingleFilePSD' objects
        self.azim_PSDs = []

        # calculate PSDs from remaining files
        for az in range(self.N_azim):

            ds_data = SMA.SingleFileTimeSeries(self.filenames[az])

            self.azim_PSDs.append(ds_data.calc_PSDs(Ndft, window, Noverlap))

        # brings some metadata to current namespace
        self.N_ch = (self.azim_PSDs[0].psd).shape[0]
        self.fs = self.azim_PSDs[0].fs
        self.df = self.azim_PSDs[0].df
        self.freq = self.azim_PSDs[0].freq


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
        broadband_SPL : (N_azim, N_ch)-shape array_like
            Integrated broadband SPL per azimuth/file, per channel, in dB re
            20 uPa RMS, within the frequency band [f_low, f_high].
        """

        self.broadband_SPL = np.zeros((self.N_azim, self.N_ch))

        for az in range(self.N_azim):
            self.azim_PSDs[az].calc_broadband_PSD(kernel_size, units)
            self.broadband_SPL[az, :] = self.azim_PSDs[az].calc_broadband_SPL(f_low, f_high)

        return self.broadband_SPL


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
        overall_SPL : (N_azim, N_ch)-shape array_like
            Integrated overall SPL per azimuth/file, per channel, in dB re 20
            uPa RMS, within the frequency band [f_low, f_high].

        """

        self.overall_SPL = np.zeros((self.N_azim, self.N_ch))

        for az in range(self.N_azim):
            self.overall_SPL[az, :] = self.azim_PSDs[az].calc_overall_SPL(f_low, f_high)

        return self.oa_SPL


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
        'peak_indices' : (N_azim, N_ch, N_peaks)-shape array_like
            Indices of all peaks above threshold.

        'peak_lims' : (N_azim, N_ch, N_peaks, 2)-shape array_like
            Lower and upper indices determining the width of each peak.
            Defined as the points where the peak in raw PSD crosses the PSD
            broadband component.
        """

        N_peaks_max = 0

        self.all_peaks = np.zeros((self.N_azim, self.N_ch, self.Ndft//2+1),
                                  dtype=int)
        self.all_peak_lims = np.zeros((self.N_azim, self.N_ch, self.Ndft//2+1, 2),
                                      dtype=int)

        for az in range(self.N_azim):
            self.azim_PSDs[az].find_all_peaks(f_low, f_high, dB_above_broadband)

            N_peaks = self.azim_PSDs[az].all_peaks.shape[1]

            self.all_peaks[az, :, :N_peaks] = self.azim_PSDs[az].all_peaks
            self.all_peak_lims[az, :, :N_peaks, :] = self.azim_PSDs[az].all_peak_lims

            N_peaks_max = np.max([N_peaks_max, N_peaks])

        # change size of 'all_peaks' to largest no. of peaks found
        temp_allpeaks = np.copy(self.all_peaks[:, :, :N_peaks_max])
        self.all_peaks = np.copy(temp_allpeaks)

        # change size of 'all_peaks' to largest no. of peaks found
        temp_allpeak_lims = np.copy(self.all_peak_lims[:, :, :N_peaks_max, :])
        self.all_peak_lims = np.copy(temp_allpeak_lims)

        return self.all_peaks, self.all_peak_lims


    def calc_peaks_SPL(self):
        """
        Returns an array of peaks' levels per azimuth/file, per channel, in
        dB re 20 uPa RMS.

        Parameters
        ----------
        None

        Returns
        -------
        peaks_SPL : (N_azim, N_ch, N_peaks)-shape array_like
            Array of integrated tones' SPL per azimuth/file, per channel, in
            dB re 20 uPa RMS.

        Notes
        -----
        Number of tones can vary per channel and azimuth/file.
        """

        N_peaks = self.all_peaks.shape[2]

        self.peaks_SPL = np.zeros((self.N_azim, self.N_ch, N_peaks))

        for az in range(self.N_azim):

            self.peaks_SPL[az, :, :] = self.azim_PSDs[az].calc_peaks_SPL()

        return self.peaks_SPL


    def calc_tonal_SPL(self):
        """
        Returns the tonal SPL per azimuth/file, per channel, as the sum of all
        tones (BPF harmonics or other peaks) SPLs.

        Parameters
        ----------
        None

        Returns
        -------
        tonal_SPL : (N_azim, N_ch)-shape array_like
            Array of integrated BPF harmonics' SPL per azimuth/file, per
            channel, in dB re 20 uPa RMS.

        Notes
        -----
        Indices are zero-based, but BPF harmonics are one-based: index 0
        represents 1xBPF, index 1 represents 2xBPF, etc.
        """

        self.tonal_SPL = np.zeros((self.N_azim, self.N_ch))

        for az in range(self.N_azim):
            self.tonal_SPL[az, :] = self.azim_PSDs[az].calc_tonal_SPL()

        return self.tonal_SPL


    def az_elev_to_polar(self, spl_name, combine_90deg=False):
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

        spl_names_list = ['oa_SPL', 'broadband_SPL', 'tonal_SPL']
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


# ##########################################################################
# %% Auxiliary functions
# ##########################################################################

def round_to_nearest_odd(x):
    " Rounds number to nearest odd integer. "
    return int(2*np.floor(x/2)+1)
