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


from sal_constants import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, DEFAULT_WINDOW

from sal_psd_utilities import MultiChannelPSD


# #############################################################################
# %% Class 'DSRawTimeSeries'
# #############################################################################

class DSRawTimeSeries:
    """
    Class to read raw measurement data from Dewesoft HDF5 files
    """

    # *************************************************************************
    def __init__(self, filename, mic_channel_names, T=30, fs=50000,
                 other_ch_names=None, fs2=None):

        self.filename = filename
        self.mic_channel_names = mic_channel_names

        # nominal duration of data recording, in seconds
        self.T = T

        # default sampling freq
        self.fs = fs

        # time vector
        self.t = np.linspace(0, self.T - 1/self.fs, self.T*self.fs)

        # 2nd sampling freq, for data acquired with SIRIUSiwe STG-M rack unit
        # (e.g. load cell, thermocouple)
        if fs2:
            self.fs2 = fs2
            self.t2 = np.linspace(0, self.T - 1/self.fs2, self.T*self.fs2)

        # read mic data from filename
        self._read_mic_chs(filename, mic_channel_names)

        # if present, read other channels' data from filename
        if other_ch_names:
            self.other_ch_names = other_ch_names
            self._read_other_chs(filename, other_ch_names)


    # *************************************************************************
    def _read_mic_chs(self, filename, mic_ch_names):
        """
        Reads microphone data from a HDF5 file generated by Dewesoft. Data
        length and sampling frequencies are defined at initialisation.

        Parameters
        ----------
        filename : str
            String containing path and filename to be read. Must be in HDF5
            format.

        mic_ch_names : list
            List of strings containing the names of the 10 microphone channels
            as set up in DewesoftX.
        """

        with h5py.File(filename, 'r') as h5file:

            # -----------------------------------------------------------------
            # check for recording length of 1st mic channel, in case actual data is
            # shorter than (T*fs) samples
            rec_length = h5file[mic_ch_names[0]].shape[0]
            assert rec_length <= self.T*self.fs, \
                "Actual hdf5 file data length is longer than 'T*fs' declared for this instance!"

            # -----------------------------------------------------------------
            # assert all mic channel names actually exist in h5file
            channel_names = list(h5file.keys())

            assert set(mic_ch_names).issubset(channel_names), \
                "Channel named {} does not exist in this hdf5 file!"

            # read number of mic channels
            self.N_ch = len(mic_ch_names)
            self.mic_data = np.zeros((self.N_ch, self.T*self.fs))

            # read mic data from HDF5 file
            for ch_index, ch_name in enumerate(mic_ch_names):
                self.mic_data[ch_index, :rec_length] = h5file[ch_name][:, 1]

            # -----------------------------------------------------------------


    # *************************************************************************
    def _read_other_chs(self, filename, other_ch_names):
        """
        Reads other channels' data from a HDF5 file generated by Dewesoft.
        Data length and sampling frequencies are defined at initialisation.

        Parameters
        ----------
        filename : str
            String containing path and filename to be read.

        other_ch_names : list
            List of strings containing the names of the other channels
            in DewesoftX - e.g. 'RPM', 'Temperature', 'LoadCell', etc.

        """

        with h5py.File(filename, 'r') as h5file:

            # assert all channel names actually exist in h5file
            channel_names = list(h5file.keys())

            assert set(other_ch_names).issubset(channel_names), \
                "Channel named {} does not exist in this hdf5 file!"

            # read data from HDF5 file, save as attribute
            for ch_name in other_ch_names:
                data = h5file[ch_name][:, 1]
                setattr(self, ch_name, data)


    # *************************************************************************
    def calc_chs_mean(self, ch_names):
        """
        Iterates over a list of channel names and calculates their mean value
        over time. Generally used for non-acoustic data - e.g. temperature,
        load cells, etc.

        For each channel named 'xx', stores the result in a new
        attribute named 'mean_xx'.

        Parameters
        ----------
        other_ch_names : list
            List of strings containing the names of the other channels
            in DewesoftX - e.g. 'RPM', 'Temperature', 'LoadCell', etc.
        """

        for name in ch_names:
            assert hasattr(self, name), \
                "Channel {} does not exist in this DSRawTimeSeries instance!".format(name)

            mean_value = np.mean( getattr(self, name))
            setattr(self, 'mean_' + name, mean_value)


    # *************************************************************************
    def filter_data(self, filter_order=3, fc=50, btype='highpass'):
        """
        Filter time-domain microphone data at given filter order, cutoff
        frequency(ies), filter type, and overwrite result over original data.
        Uses Butterworth filter topology, and applies fwd-bkwd filtering.
        """

        my_filter = ss.butter(filter_order, fc, btype,
                              output='sos', fs=self.fs)

        for ch in range(self.N_ch):
            # fwd-bkwd filtering of the signals
            hp_data = ss.sosfiltfilt(my_filter, self.mic_data[ch, :])

            # overwrite original data
            self.mic_data[ch, :] = hp_data


    # *************************************************************************
    def estimate_peak_freq(self, f_low, f_high, Ndft=2**14):
        """
        Estimates the centre frequency of the tallest peak in the spectrum
        within a given frequency range [f_low, f_high] (in Hz). Estimate is
        averaged across all channels.

        Uses a "spectral centroid" calculation, so it can estimate values in
        between frequency samples.
        """

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        df = self.fs/Ndft
        freq = np.linspace(0, self.fs - df, Ndft)[:Ndft//2+1]

        PSDs = self.calc_PSDs(Ndft, window='hann', Noverlap=Ndft//2)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # calculate broadband component of PSD to use as amplitude threshold
        median_kernel_Hz = 100      # [Hz]
        PSDs.calc_broadband_PSD(median_kernel_Hz, units='Hz')

        # select freqs between f_low and f_high
        freq_mask = (freq >= f_low) & (freq <= f_high)

        # freq index of first mask entry
        mask_index = np.argwhere(freq_mask == 1)[0][0]

        f_peak = np.zeros(PSDs.N_ch)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        for ch in range(PSDs.N_ch):

            # find tallest peak within freq range
            peak_index, peak_properties = ss.find_peaks(PSDs.psd[ch, freq_mask],
                                                        height=PSDs.psd_broadband[ch, freq_mask])

            # -----------------------------------------------------------------
            # if no peaks were found, write NaN in peak freq
            if (peak_properties['peak_heights']).size == 0:
                f_peak[ch] = np.nan

            # -----------------------------------------------------------------
            # if one or more peaks were found....
            else:
                n_tallest = np.argmax(peak_properties['peak_heights'])

                # list of indices for peak freq over all channels
                n_peak = mask_index + peak_index[n_tallest]

                # calculate spectral centroid around tallest peak to improve
                # estimate of peak frequency
                search_radius = 2
                fpeak_mask = np.arange(n_peak - search_radius,
                                       n_peak + search_radius + 1)

                f_peak[ch] = _calc_spectral_centroid(PSDs.freq[fpeak_mask],
                                                     PSDs.psd[ch, fpeak_mask])
            # -----------------------------------------------------------------

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        return np.nanmean(f_peak)


    # *************************************************************************
    def calc_PSDs(self, Ndft=DEFAULT_NDFT, Noverlap=DEFAULT_NOVERLAP,
                  window=DEFAULT_WINDOW, t0=0):
        """
        Calculates and outputs the PSDs of all channels. Optionally, skip
        initial segment 't0'.
        """

        n = t0*self.fs

        PSDs = np.zeros((self.N_ch, Ndft//2+1))
        for ch in range(self.N_ch):
            freq, PSDs[ch, :] = ss.welch(self.mic_data[ch, n:], self.fs,
                                         window=window, nperseg=Ndft,
                                         noverlap=Noverlap)

        myPSDs = MultiChannelPSD(PSDs, freq, self.fs, Ndft, Noverlap, window)

        return myPSDs


    # *************************************************************************
    def export_wavs(self, wav_filename, channels=10, subtype='FLOAT'):
        """
        Exports current 'mic_data' time series as a multichannel .WAV file.
        Requires 'soundfile' (previously 'pysoundfile') package.

        Parameters
        ----------
        wav_filename : string
            File name of multichannel .wav file.

        channels : int or list, optional
            Channels to output. If 'int', outputs this many channels in
            ascending order; if list, output channel values contained in list,
            in the given order.

        subtype : string, optional
            String defining .wav file subtype. Use
            'soundfile.available_subtypes()' to list current options. Default
            value is 'FLOAT' for 32-bit float.

        Returns
        -------
        None.

        Notes
        -----
        Maximum value allowed in 'channels' is 10.

        If 'channel=8', the output will be a 8-channel .wav file containing
        data from mics index 0 to 7.

        If 'channels=[2, 6, 5]', the output will be a 3-channel .wav file
        containing data from mics 2, 6 and 5, in that order.
        """

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # check all mic values are <=1, print warning if not
        if not (np.abs(self.mic_data)<=1).all():
            print("WARNING: Some microphone signal amplitudes are above unity!")

        # checks filename ends in '.wav' extension, add if it doesn't
        if wav_filename[-4:] != '.wav':
            wav_filename += '.wav'

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # check whether 'channels' is int, if so create channel list
        if isinstance(channels, int):
            assert channels<=10, \
                "If int, 'channels' must be equal to or less than 10!"
            ch_list = [n for n in range(channels)]

        # if channels is list/np array, copy as is
        elif isinstance(channels, (list, np.ndarray)):
            assert all(ch<10 for ch in channels), \
                "If list, channel indices must be less than 10!"
            ch_list = channels.copy()

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # write .wav file, up to 'n_channels'
        sf.write(wav_filename, self.mic_data[ch_list].T, self.fs,
                 subtype)
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


# #############################################################################
# %% Auxiliary functions
# #############################################################################

def _calc_spectral_centroid(x, y):
    """
    Calculates the spectral centroid for two arrays 'x' and 'y', where 'y'
    is assumed to be a function of 'x'.

    This function is meant to be used for roughly estimating tone centre
    frequency when these fall in between frequency samples.
    """
    return np.sum(x*y)/np.sum(y)


# #############################################################################
def calc_ac_power(SPL_polar, polar_angles, R, rho0=1.22, c0=340,
                  Pow_ref=1e-12, angle_units='deg'):
    """
    Calculates the radiated sound power level Lp by integrating a SPL
    directivity over elev=[0, 180]. Assumes axial symmetry.

    Parameters
    ----------
    SPL_polar : (N_angles,)-shape array_like
        Array of Sound Pressure Levels, in dB re 20 uPa.

    polar_angles : (N_angles,)-shape array_like
        Array of polar angles over which SPLs are defined. Should cover
        0 to 180deg / pi rad.

    R : float
        Radius at which SPLs were measured/calculated, in metres.

    rho0 : float, optional
        Air density, in kg/m^3. The default is 1.22 kg/m^3.

    c0 : float, optional
        Speed of sound, in metres/second. The default is 340 m/s.

    Pow_ref : float, optional
        Reference acoustic power, in Watts. The default is 1e-12 W.

    angle_units : {'deg', 'rad'}, optional
        String defining angle units. The default is 'deg'.

    Returns
    -------
    Lp : float
        Radiated acoustic power level Lp, in dB ref 1e-12 W.
    """

    # *************************************************************************
    assert angle_units in ['deg', 'rad'],\
        "Input argument 'angle_units' must be either 'deg' or 'rad' !"

    if angle_units == 'deg':
        polar_angles_rad = polar_angles*np.pi/180
    elif angle_units == 'rad':
        polar_angles_rad = polar_angles

    # *************************************************************************
    # convert dB ref 20 uPa to squared pressures
    p2_polar = (P_REF**2)*(10**(SPL_polar/10))

    # integrand function over polar angle
    integrand = (p2_polar*(R**2)/(rho0*c0))*np.sin(polar_angles_rad)

    Pw = 2*np.pi*np.trapz(integrand, polar_angles_rad)

    return 10*np.log10(Pw/Pow_ref)
