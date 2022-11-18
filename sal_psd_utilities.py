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


from sal_constants import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, DEFAULT_WINDOW


# #############################################################################
# %% Class 'MultiChannelPSD'
# #############################################################################

class MultiChannelPSD:
    """
    Class to store multi-channel PSD and associated frequency-domain
    information. PSDs are assumed single-sided.
    """

    def __init__(self, psd, freq, fs, Ndft=DEFAULT_NDFT,
                 Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):

        # Power spectral density values (single sided)
        #   (N_mics, Ndft//2+1)-shape array_like
        self.psd = np.atleast_2d(psd)

        # number of microphone channels
        self.N_mics = self.psd.shape[0]

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
            "Cannot find peaks: PSD instance does not have 'psd_broadband' attribute defined!"

        gain_above_broadband = 10**(dB_above_broadband/10)

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        # initialize 'peak_indices' with a large size (Ndft/2+1), reduce it later
        self.peak_indices = np.zeros((self.N_ch, self.Ndft//2+1), dtype=int)

        N_peaks = 0

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
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

        # change size of 'peak_indices' to largest no. of peaks found
        temp_peaks = np.copy(self.peak_indices[:, :N_peaks])
        self.peak_indices = np.copy(temp_peaks)

        # add initial index of freq_mask to all non-zero entries
        self.peak_indices[self.peak_indices!=0] += np.argwhere(freq_mask)[0, 0]

        # replace zeros with '-1' as flag for 'no peak found'
        self.peak_indices[self.peak_indices==0] = -1

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # find peak limits
        self.peak_lims = self._find_peak_lims(self.peak_indices)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

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

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        integrated_broadband_psd = np.sum(self.psd_broadband[:, freq_mask],
                                          axis=1)*self.df

        self.broadband_SPL = 10*np.log10(integrated_broadband_psd/(P_REF**2))

        return self.broadband_SPL


    # *************************************************************************
    def calc_oa_SPL(self, f_low, f_high):
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
        oa_SPL : (N_ch,)-shape array_like
            Integrated overall SPL per channel, in dB re 20 uPa RMS, within
            the frequency band [f_low, f_high].
        """

        freq_mask = (self.freq >= f_low) & (self.freq <= f_high)

        integrated_oa_psd = np.sum(self.psd[:, freq_mask], axis=1)*self.df

        self.oa_SPL = 10*np.log10(integrated_oa_psd/(P_REF**2))

        return self.oa_SPL


    # *************************************************************************
    def calc_all_peaks_SPL(self):
        """
        Returns array of all tones' levels per channel, in dB re 20 uPa RMS.

        Parameters
        ----------
        None

        Returns
        -------
        all_peaks_SPL : (N_ch, N_peaks)-shape array_like
            Array of integrated tones' SPL per channel, in dB re 20 uPa RMS.

        Notes
        -----
        Number of tones can vary across channels.
        """

        assert hasattr(self, 'all_peak_lims'), "Attribute 'all_peak_lims' not found!"

        N_peaks = self.all_peaks.shape[1]

        self.all_peaks_SPL = np.zeros((self.N_ch, N_peaks))

        for ch in range(self.N_ch):
            for n_pk in range(N_peaks):

                # if peak lims is -1 (no peak found), SPL is NaN
                if self.all_peak_lims[ch, n_pk, 0] == -1:
                    self.all_peaks_SPL[ch, n_pk] = np.nan

                else:
                    peak_range = np.arange(self.all_peak_lims[ch, n_pk, 0],
                                           self.all_peak_lims[ch, n_pk, 1]+1)

                    # subtract broadband content from PSD peak
                    peak_minus_bb = (self.psd[ch, peak_range]
                                     - self.psd_broadband[ch, peak_range])

                    integrated_peak_psd = np.sum(peak_minus_bb)*self.df

                    self.all_peaks_SPL[ch, n_pk] = 10*np.log10(integrated_peak_psd/(P_REF**2))

        return self.all_peaks_SPL


    # *************************************************************************
    def calc_tonal_SPL(self):
        """
        Returns the tonal SPL per channel, as the sum of all tonal (BPF
        harmonics or other peaks) SPLs.

        Parameters
        ----------
        None

        Returns
        -------
        tonal_SPL : (N_ch,)-shape array_like
            Array of integrated BPF harmonics' SPL per channel, in dB re 20 uPa
            RMS.

        Notes
        -----
        Indices are zero-based, but BPF harmonics are one-based: index 0
        represents 1xBPF, index 1 represents 2xBPF, etc.
        """

        assert (hasattr(self, 'bpf_peak_lims') or hasattr(self, 'all_peak_lims')),\
            "Attributes 'bpf_peak_lims' or 'all_peak_lims' not found!"

        self.tonal_SPL = np.zeros(self.N_ch)

        # If analysing BPF peaks only:
        if hasattr(self, 'bpf_peak_lims'):
            self.calc_bpf_SPL()
            peaks_SPL = self.bpf_SPL

        # otherwise, if analysing all peaks in PSD:
        elif hasattr(self, 'all_peak_lims'):
            self.calc_all_peaks_SPL()
            peaks_SPL = self.all_peaks_SPL


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


# class MultiFilePSDs:
#     """
#     Class to store post-processed frequency-domain power spectral density data
#     from multiple files, collected over a list of azimuthal angles.
#     """

#     def __init__(self, filenames, N_blades, azim_angles, nominal_rpm,
#                  Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):

#         self.filenames = filenames

#         self.N_blades = N_blades

#         self.azim_angles = azim_angles
#         self.N_azim = len(azim_angles)

#         self.nominal_rpm = nominal_rpm

#         assert len(self.filenames) == self.N_azim, "Number of elements in 'filenames' and 'azim_angles' does not match!"

#         self.Ndft = Ndft
#         self.window = window
#         self.Noverlap = Noverlap

#         self.thrust_azim = np.zeros(self.N_azim)
#         self.temp_azim = np.zeros(self.N_azim)
#         self.rpm_azim = np.zeros(self.N_azim)
#         self.bpf_azim = np.zeros(self.N_azim)

#         self.calc_azim_PSDs(Ndft, window, Noverlap)


#     def calc_azim_PSDs(self, Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP,
#                        window=DEFAULT_WINDOW):
#         """
#         Calculates PSDs for all files in list of file names.
#         """

#         # create list of 'MultiChannelPSD' objects
#         self.azim_PSDs = []

#         # calculate PSDs from remaining files
#         for az in range(self.N_azim):

#             ds_data = ds_mic_arc.DSRawTimeSeries(self.filenames[az],
#                                                  self.N_blades)

#             self.azim_PSDs.append(ds_data.calc_PSDs(Ndft, window, Noverlap))

#             if hasattr(ds_data, 'avg_rpm'):
#                 # If time-domain data has attr 'avg_rpm', use that
#                 self.rpm_azim[az] = ds_data.avg_rpm
#             else:
#                 # if not, get nominal RPM and use it to estimate RPM and BPF
#                 # from f_shaft peak in acoustic data

#                 f_shaft_approx = self.nominal_rpm/60
#                 f_shaft = ds_data.estimate_peak_freq(f_shaft_approx - 20,
#                                                      f_shaft_approx + 20,
#                                                      Ndft=2**14)
#                 self.rpm_azim[az] = f_shaft*60
#                 self.bpf_azim[az] = f_shaft*self.N_blades

#             if hasattr(ds_data, 'avg_thrust'):
#                 self.thrust_azim[az] = ds_data.avg_thrust

#             if hasattr(ds_data, 'avg_temp'):
#                 self.temp_azim[az] = ds_data.avg_temp

#             if hasattr(ds_data, 'avg_bpf'):
#                 self.bpf_azim[az] = ds_data.avg_bpf

#         # brings some metadata to current namespace
#         self.N_ch = (self.azim_PSDs[0].psd).shape[0]
#         self.fs = self.azim_PSDs[0].fs
#         self.df = self.azim_PSDs[0].df
#         self.freq = self.azim_PSDs[0].freq



#     def calc_broadband_SPL(self, f_low, f_high, f_type='Hz'):
#         """
#         Calculates the broadband SPL within a frequency band ['f_low', 'f_high']
#         by integrating the broadband PSDs within this band. Results are returned
#         in dB re 20 uPa RMS.


#         Parameters
#         ----------
#         f_low : float
#             Low frequency limit, in Hz.

#         f_high : float
#             High frequency limit, in Hz.

#         'f_type' : {'Hz', 'f_shaft'}, optional
#             String determining whether 'f_low', 'f_high' are absolute
#             frequencies in Hz or ratios of the shaft frequency. Default is 'Hz'


#         Returns
#         -------
#         broadband_SPL : (N_azim, N_ch)-shape array_like
#             Integrated broadband SPL per azimuth/file, per channel, in dB re
#             20 uPa RMS, within the frequency band [f_low, f_high].
#         """

#         assert f_type in ['Hz', 'f_shaft'], \
#             "Input argument 'f_type' must be either 'Hz' or 'f_shaft' !"

#         self.broadband_SPL = np.zeros((self.N_azim, self.N_ch))

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         if f_type == 'Hz':

#             for az in range(self.N_azim):
#                 self.azim_PSDs[az].calc_broadband_PSD()
#                 self.broadband_SPL[az, :] = self.azim_PSDs[az].calc_broadband_SPL(f_low, f_high)

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         elif f_type == 'f_shaft':
#             for az in range(self.N_azim):

#                 f_shaft = self.rpm_azim[az]/60

#                 self.azim_PSDs[az].calc_broadband_PSD()
#                 self.broadband_SPL[az, :] = self.azim_PSDs[az].calc_broadband_SPL(f_low*f_shaft, f_high*f_shaft)

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#         return self.broadband_SPL


#     def calc_oa_SPL(self, f_low, f_high, f_type='Hz'):
#         """
#         Calculates the overall SPL within a frequency band ['f_low', 'f_high']
#         by integrating the PSDs within this band. Results are returned
#         in dB re 20 uPa RMS.

#         Parameters
#         ----------
#         f_low : float
#             Low frequency limit, in Hz.

#         f_high : float
#             High frequency limit, in Hz.

#         'f_type' : {'Hz', 'f_shaft'}
#             String determining whether 'f_low', 'f_high' are absolute
#             frequencies in Hz or ratios of the shaft frequency.

#         Returns
#         -------
#         oa_SPL : (N_azim, N_ch)-shape array_like
#             Integrated overall SPL per azimuth/file, per channel, in dB re 20
#             uPa RMS, within the frequency band [f_low, f_high].

#         """

#         assert f_type in ['Hz', 'f_shaft'], \
#             "Input argument 'f_type' must be either 'Hz' or 'f_shaft' !"

#         self.oa_SPL = np.zeros((self.N_azim, self.N_ch))

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         if f_type == 'Hz':

#             for az in range(self.N_azim):
#                 self.oa_SPL[az, :] = self.azim_PSDs[az].calc_oa_SPL(f_low, f_high)

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         elif f_type == 'f_shaft':
#             for az in range(self.N_azim):

#                 f_shaft = self.rpm_azim[az]/60
#                 self.oa_SPL[az, :] = self.azim_PSDs[az].calc_oa_SPL(f_low*f_shaft, f_high*f_shaft)

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#         return self.oa_SPL


#     def find_bpf_peaks(self, N_peaks=10, dB_above_broadband=0, radius=10,
#                        units='points'):
#         """
#         Finds peaks in PSDs for first 'N_peaks' Blade Passing Frequency
#         harmonics, with attribute 'bpf' being the nominal BPF in Hz. Optional
#         arguments are the height above PSD broadband component as threshold,
#         and maximum search radius around nominal frequency.

#         Parameters
#         ----------
#         N_peaks = int, optional
#             Number of peaks to search for. Default is 10.

#         dB_above_broadband : float, optional
#             Minimum peak height above broadband PSD component, in decibels.

#         radius : int or float, optional
#             Search radius around nominal frequency, in points (int) or Hz
#             (float). Default is 10 points.

#         units : {'points', 'Hz'}, optional
#             Units for peak search radius. Default is 'points'.


#         Returns
#         -------
#         'bpf_peaks' : (N_azim, N_ch, N_peaks)-shape array_like
#             Indices of BPF peaks.

#         'bpf_peak_lims' : (N_azim, N_ch, N_peaks, 2)-shape array_like
#             Lower and upper indices determining the width of each peak.
#             Defined as the points where the peak in raw PSD crosses the PSD
#             broadband component.
#         """

#         assert units in ['points', 'Hz'], "Unknown input for 'units' - must be 'points' or 'Hz' !"

#         # if kernel size is given in Hz, calculate equivalent length in points
#         if units == 'Hz':
#             radius_Hz = np.copy(radius)
#             radius = round_to_nearest_odd(radius_Hz/self.df)


#         self.bpf_peaks = np.zeros((self.N_azim, self.N_ch, N_peaks),
#                                   dtype=int)
#         self.bpf_peak_lims = np.zeros((self.N_azim, self.N_ch, N_peaks, 2),
#                                       dtype=int)

#         for az in range(self.N_azim):
#             self.azim_PSDs[az].find_bpf_peaks(self.bpf_azim[az], N_peaks,
#                                               dB_above_broadband, radius, units)

#             self.bpf_peaks[az, :, :] = self.azim_PSDs[az].bpf_peaks
#             self.bpf_peak_lims[az, :, :, :] = self.azim_PSDs[az].bpf_peak_lims

#         return self.bpf_peaks, self.bpf_peak_lims


#     def find_all_peaks(self, f_low, f_high, dB_above_broadband=3):
#         """
#         Finds all peaks in PSD spectrum within a bandwidth [f_low, f_high]. Peaks
#         are not restricted to be at the BPF harmonics. Optional arguments are the
#         height above PSD broadband component as threshold.

#         Parameters
#         ----------
#         f_low : float
#             Low frequency limit, in Hz.

#         f_high : float
#             High frequency limit, in Hz.

#         dB_above_broadband : float, optional
#             Minimum peak height above broadband PSD component, in decibels. Default
#             value is 3 dB.


#         Returns
#         -------
#         'all_peaks' : (N_azim, N_ch, N_peaks)-shape array_like
#             Indices of all peaks above threshold.

#         'all_peak_lims' : (N_azim, N_ch, N_peaks, 2)-shape array_like
#             Lower and upper indices determining the width of each peak.
#             Defined as the points where the peak in raw PSD crosses the PSD
#             broadband component.
#         """

#         N_peaks_max = 0

#         self.all_peaks = np.zeros((self.N_azim, self.N_ch, self.Ndft//2+1),
#                                   dtype=int)
#         self.all_peak_lims = np.zeros((self.N_azim, self.N_ch, self.Ndft//2+1, 2),
#                                       dtype=int)

#         for az in range(self.N_azim):
#             self.azim_PSDs[az].find_all_peaks(f_low, f_high, dB_above_broadband)

#             N_peaks = self.azim_PSDs[az].all_peaks.shape[1]

#             self.all_peaks[az, :, :N_peaks] = self.azim_PSDs[az].all_peaks
#             self.all_peak_lims[az, :, :N_peaks, :] = self.azim_PSDs[az].all_peak_lims

#             N_peaks_max = np.max([N_peaks_max, N_peaks])

#         # change size of 'all_peaks' to largest no. of peaks found
#         temp_allpeaks = np.copy(self.all_peaks[:, :, :N_peaks_max])
#         self.all_peaks = np.copy(temp_allpeaks)

#         # change size of 'all_peaks' to largest no. of peaks found
#         temp_allpeak_lims = np.copy(self.all_peak_lims[:, :, :N_peaks_max, :])
#         self.all_peak_lims = np.copy(temp_allpeak_lims)

#         return self.all_peaks, self.all_peak_lims


#     def calc_bpf_SPL(self):
#         """
#         Returns an array of BPF harmonics' levels per azimuth/file, per
#         channel, in dB re 20 uPa RMS.


#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         bpf_SPL : (N_azim, N_ch, N_harms)-shape array_like
#             Array of integrated BPF harmonics' SPL per azimuth/file, per
#             channel, in dB re 20 uPa RMS.

#         Notes
#         -----
#         Indices are zero-based, but BPF harmonics are one-based: index 0
#         represents 1xBPF, index 1 represents 2xBPF, etc.
#         """

#         N_harms = self.bpf_peaks.shape[2]

#         self.bpf_SPL = np.zeros((self.N_azim, self.N_ch, N_harms))

#         for az in range(self.N_azim):

#             self.bpf_SPL[az, :, :] = self.azim_PSDs[az].calc_bpf_SPL()

#         return self.bpf_SPL


#     def calc_all_peaks_SPL(self):
#         """
#         Returns an array of all tones' levels per azimuth/file, per
#         channel, in dB re 20 uPa RMS.


#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         all_peaks_SPL : (N_azim, N_ch, N_peaks)-shape array_like
#             Array of integrated tones' SPL per azimuth/file, per channel, in
#             dB re 20 uPa RMS.

#         Notes
#         -----
#         Number of tones can vary per channel and azimuth/file.
#         """

#         N_peaks = self.all_peaks.shape[2]

#         self.all_peaks_SPL = np.zeros((self.N_azim, self.N_ch, N_peaks))

#         for az in range(self.N_azim):

#             self.all_peaks_SPL[az, :, :] = self.azim_PSDs[az].calc_all_peaks_SPL()

#         return self.all_peaks_SPL


#     def calc_tonal_SPL(self):
#         """
#         Returns the tonal SPL per azimuth/file, per channel, as the sum of all
#         tones (BPF harmonics or other peaks) SPLs.

#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         tonal_SPL : (N_azim, N_ch)-shape array_like
#             Array of integrated BPF harmonics' SPL per azimuth/file, per
#             channel, in dB re 20 uPa RMS.

#         Notes
#         -----
#         Indices are zero-based, but BPF harmonics are one-based: index 0
#         represents 1xBPF, index 1 represents 2xBPF, etc.
#         """

#         self.tonal_SPL = np.zeros((self.N_azim, self.N_ch))

#         for az in range(self.N_azim):
#             self.tonal_SPL[az, :] = self.azim_PSDs[az].calc_tonal_SPL()

#         return self.tonal_SPL


#     def az_elev_to_polar(self, spl_name, combine_90deg=False):
#         """
#         Calculates the polar directivity pattern for a given SPL calculation
#         by combining measurements made at (azim, elev) = (0, 0-90) and
#         (azim, elev) = (180, 0-90).

#         Parameters
#         ----------
#         spl_name : {'oa_SPL', 'broadband_SPL', 'tonal_SPL',
#                     'bpf1_SPL', 'bpf2_SPL'}, str
#             String containing name of SPL function to output.

#         combine_90deg : bool, optional
#             Boolean flag denoting whether to combine measurements at 90deg.
#             See Notes below.


#         Returns
#         -------
#         polar_angles : (N_angles,)-shape array
#             Array containing the elevation angles from 0deg to 180deg. See
#             Notes below on how 'combine_90deg' flag affects this array.

#         SPL_polar : (N_angles,)-shape array
#             Array containing the SPL values over elevation angles. See Notes
#             below on how 'combine_90deg' flag affects this array.

#         Notes
#         -----
#         A range of polar angles [0 - 180deg] is created by combining
#         measurements made at 0deg azim and 180deg azim, but the two overlapping
#         90deg elevation readings are dealt with in one of two ways:

#             - if 'combine_90deg = True', they are averaged and shown as a
#             single value at 90deg. The resulting arrays will have 'N_angles=19'
#             points equally spaced between 0 and 180deg elevation.

#             - if 'combine_90deg = False', they are are instead shown separately
#             at 89deg and 91deg, in order to assess any differences between
#             them. The resulting arrays will have 'N_angles=20' points between
#             0 and 180deg elevation, and all except 90deg are equally spaced.
#         """

#         spl_names_list = ['oa_SPL', 'broadband_SPL', 'tonal_SPL',
#                           'bpf1_SPL', 'bpf2_SPL', 'bpf3_SPL']

#         assert spl_name in spl_names_list, "'spl_name' not recognized!"

#         if spl_name in ['oa_SPL', 'broadband_SPL', 'tonal_SPL']:
#             SPL_az_elev = getattr(self, spl_name)

#         elif spl_name == 'bpf1_SPL':
#             SPL_az_elev = getattr(self, 'bpf_SPL')[:, :, 0]

#         elif spl_name == 'bpf2_SPL':
#             SPL_az_elev = getattr(self, 'bpf_SPL')[:, :, 1]

#         elif spl_name == 'bpf3_SPL':
#             SPL_az_elev = getattr(self, 'bpf_SPL')[:, :, 2]


#         if combine_90deg:
#             polar_angles = np.linspace(0, 180, 19)
#             SPL_polar = np.zeros((19))

#             # write 0deg-80deg (0deg azim in SPL_az_elev)
#             SPL_polar[:9] = SPL_az_elev[0, :9]

#             # write 100deg-180deg (180deg azim in SPL_az_elev, reversed)
#             SPL_polar[10:] = (SPL_az_elev[-1, :9])[::-1]

#             # average 90deg elev value from 0deg azim and 180deg azim
#             SPL_polar[9] = np.mean([SPL_az_elev[0, 9], SPL_az_elev[-1, 9]])


#         else:
#             polar_angles_1 = np.linspace(0, 90, 10)
#             polar_angles_1[-1] = 89

#             polar_angles_2 = np.linspace(90, 180, 10)
#             polar_angles_2[0] = 91

#             polar_angles = np.concatenate((polar_angles_1, polar_angles_2))

#             SPL_polar = np.zeros((20))

#             # write 0deg-89deg (0deg azim in SPL_full)
#             SPL_polar[:10] = SPL_az_elev[0, :]

#             # write 91deg-180deg (180deg azim in SPL_full, reversed)
#             SPL_polar[10:] = (SPL_az_elev[-1, :])[::-1]


#         return polar_angles, SPL_polar


#     def export_directivity(self, filename):
#         """
#         Export directivity data as HDF5 file. The data contains the
#         measured one-sided PSD over (azim, elev) directions, plus a reference
#         acoustic pressure time series for the inlet direction.

#         The exported HDF5 file will use the following variable (or
#         "dataset", in HDF5 terms) names, with the following attributes:

#             ac_pressure_inlet : (Nt,)-shape array
#                 azim_inlet : float
#                 elev_inlet : float

#             psd : (N_azim, N_elev, Ndft//2+1)-shape array

#             rpm_azim : (N_azim,)-shape array
#                 nominal_rpm : float

#             azim_angles : (N_azim)-shape array
#                 units : str
#             elev_angles : (N_elev)-shape array
#                 units : str

#             freq : (Ndft//2+1,)-shape array
#                 Ndft : int
#                 fs : float
#         """

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         # re-reads file, extracts acoustic pressure time series at inlet
#         # direction
#         ds_data = ds_mic_arc.DSRawTimeSeries(self.filenames[-1], self.N_blades)
#         p_inlet = ds_data.mic_data[0, :]

#         # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#         psd = np.zeros((self.N_azim, self.N_ch, self.Ndft//2+1,))

#         elev_angles = np.linspace(0, 90, self.N_ch)

#         # create new file, fail if file already exists
#         with h5py.File(filename + '.h5', 'x') as dir_file:

#             # ----------------------------------------------------------------
#             # acoustic pressure at inlet direction
#             h5p_inlet = dir_file.create_dataset('p_inlet', data=p_inlet)
#             h5p_inlet.attrs['azim_inlet'] = 180.
#             h5p_inlet.attrs['elev_inlet'] = 0.

#             # ----------------------------------------------------------------
#             # psd as function of (freq, azim, elev)
#             h5psd = dir_file.create_dataset('psd', data=psd)
#             for az in range(self.N_azim):
#                 h5psd[az, :, :] = self.azim_PSDs[az].psd

#             # ----------------------------------------------------------------
#             # Measured RPM value at each azim angle, plus nominal RPM
#             h5rpm = dir_file.create_dataset('rpm_azim',
#                                             data = self.rpm_azim)
#             h5rpm.attrs['nominal_rpm'] = self.nominal_rpm

#             # ----------------------------------------------------------------
#             # azimuth angles, in degrees
#             h5azim = dir_file.create_dataset('azim_angles',
#                                              data = self.azim_angles)
#             h5azim.attrs['units'] = 'degrees'

#             # elevation angles, in degrees
#             h5elev = dir_file.create_dataset('elev_angles',
#                                              data = elev_angles)
#             h5elev.attrs['units'] = 'degrees'

#             # ----------------------------------------------------------------
#             # Frequency values, plus Ndft and sampling freq
#             h5freq = dir_file.create_dataset('freq', data = self.freq)
#             h5freq.attrs['Ndft'] = self.Ndft
#             h5freq.attrs['fs'] = self.fs



# ##########################################################################
# %% Auxiliary functions
# ##########################################################################

def round_to_nearest_odd(x):
    " Rounds number to nearest odd integer. "
    return int(2*np.floor(x/2)+1)
