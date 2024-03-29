"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono


Classes and functions to process a single HDF5 file from Dewesoft.

Author:
    Fabio Casagrande Hirono
    November 2022
"""

import h5py
import numpy as np

import soundfile as sf
import scipy.signal as ss

import salford_mic_arc.sma_angular_resampling as AR

from salford_mic_arc.sma_consts_aux import P_REF, DEFAULT_NDFT, DEFAULT_NOVERLAP, \
    DEFAULT_WINDOW, _calc_centroid, _round_to_nearest_odd


# #############################################################################
# %% Class 'InputFile' and 'InputFiles'
# #############################################################################

class InputFile:
    """
    Class to hold Dewesoft HDF5 file metadata
    """
    def __init__(self, filename):
        # name of file to be read (must be HDF5)
        #   str
        assert isinstance(filename, str), "'filename' is not str!"
        self.filename = filename
        self.is_rotor = False


    def set_mic_channel_names(self, mic_channel_names):
        # list of microphone channels' names in 'filename'
        assert isinstance(mic_channel_names, list), "'mic_channel_names' is not list!"
        self.mic_channel_names = mic_channel_names


    def set_other_ch_names(self, other_ch_names):
        # list of non-acoustic channels in 'filename'
        assert isinstance(other_ch_names, list), "'other_ch_names' is not list!"
        self.other_ch_names = other_ch_names

    def set_recording_length(self, T):
        # nominal duration of data recording, in seconds
        #   float
        assert isinstance(T, (float, int)), "'T' is not float or int!"
        self.T = T

    def set_sampling_freq(self, fs):
        # default sampling freq
        #   float
        assert isinstance(fs, (float, int)), "'fs' is not float or int!"
        self.fs = fs

    def set_sampling_freq2(self, fs2):
        # 2nd sampling freq, for data acquired with SIRIUSiwe STG-M rack unit
        # (e.g. load cell, thermocouple)
        #   float
        assert isinstance(fs2, (float, int)), "'fs2' is not float or int!"
        self.fs2 = fs2

    # *************************************************************************
    # For measurements of rotating devices (rotors, propellers, fans, etc)

    def set_is_rotor(self, is_rotor):
        # For measurements using rotating devices: bool
        assert isinstance(is_rotor, bool), "'is_rotor' is not boolean!"
        self.is_rotor = is_rotor

    def set_N_blades(self, N_blades):
        # For measurements using rotating devices: number of rotor blades
        #   int
        assert isinstance(N_blades, int), "'N_blades' is not int!"
        self.N_blades = N_blades

    def set_R_blades(self, R_blades):
        # For measurements using rotating devices: rotor blades' radius [m]
        #   float
        assert isinstance(R_blades, (float, int)), "'R_blades' is not float or int!"
        self.R_blades = R_blades

    def set_rpm_attr_name(self, rpm_attr_name):
        # For measurements using rotating devices: name of attribute containing
        # measured/estimated RPM in recording
        #   str
        assert isinstance(rpm_attr_name, str), "'rpm_attr_name' is not string!"
        self.rpm_attr_name = rpm_attr_name

    # *************************************************************************


# #############################################################################
class InputFiles(InputFile):
    """
    Child class for holding metadata on multiple files
    """
    def __init__(self, filenames):
        # list of filenames to be read (must be HDF5)
        assert isinstance(filenames, list), "'filenames' is not list!"
        self.filenames = filenames
        self.is_rotor = False


    def get_InputFile(self, i):
        """
        Return 'InputFile' instance of 'i'-th filename
        """

        singlefile = InputFile(self.filenames[i])

        # update dict representation of 'InputFile' instance
        singlefile.__dict__.update(self.__dict__)

        return singlefile


# #############################################################################
# %% Class 'SingleFileTimeSeries'
# #############################################################################

class SingleFileTimeSeries:
    """
    Class to read raw measurement data from input file
    """

    # *************************************************************************
    def __init__(self, input_file):

        assert isinstance(input_file, InputFile), "Input argument is not instance of 'InputFile'!"

        # name of file to be read (must be HDF5)
        self.filename = input_file.filename
        self.is_rotor = input_file.is_rotor

        # list of microphone channels' names in 'filename'
        self.mic_channel_names = input_file.mic_channel_names
        self.N_ch = len(self.mic_channel_names)

        # nominal duration of data recording, in seconds
        #   float
        self.T = input_file.T

        # default sampling freq
        #   float
        self.fs = input_file.fs

        # time vector
        #   (T*fs,) array
        self.t = np.linspace(0, self.T - 1/self.fs, self.T*self.fs)

        # 2nd sampling freq, for data acquired with SIRIUSiwe STG-M rack unit
        # (e.g. load cell, thermocouple)
        if hasattr(input_file, 'fs2'):
            #   float
            self.fs2 = input_file.fs2

            #   (T*fs2,) array
            self.t2 = np.linspace(0, self.T - 1/self.fs2, self.T*self.fs2)

        # read mic data from filename
        self._read_mic_chs(self.filename, self.mic_channel_names)

        # if present, read other channels' data from 'filename' and calculate
        # their mean values
        if hasattr(input_file, 'other_ch_names'):
            # list of non-acoustic channels in 'filename'
            self.other_ch_names = input_file.other_ch_names
            self._read_other_chs(self.filename, self.other_ch_names)
            self.calc_channel_mean(self.other_ch_names)

        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # check if input data is a rotor-type (e.g. propeller, fan) measurement
        if input_file.is_rotor:

            self.is_rotor = True

            # Number of rotor blades
            #   int
            self.N_blades = input_file.N_blades

            # Radius of rotor blades [m]
            #   float
            self.R_blades = input_file.R_blades

            # Name of attribute containing RPM value
            #   if None, RPM must be set manually using 'set_RPM'
            if hasattr(input_file, 'rpm_attr_name'):
                self.set_RPM(getattr(self, input_file.rpm_attr_name))
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

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

            for ch_name in other_ch_names:
                assert (ch_name in channel_names), \
                    "Channel named '{}' does not exist in '{}' file!".format(ch_name, filename)

            # read data from HDF5 file, save as attribute
            for ch_name in other_ch_names:
                data = h5file[ch_name][:, 1]
                setattr(self, ch_name, data)


    # *************************************************************************
    def calc_channel_mean(self, ch_names):
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
                "Channel {} does not exist in this SingleFileTimeSeries instance!".format(name)

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

                f_peak[ch] = _calc_centroid(PSDs.freq[fpeak_mask],
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

        myPSDs = SingleFilePSD(self.filename, PSDs, freq, self.fs, Ndft,
                               Noverlap, window)

        # if file is a rotor measurement, copy rotor-related attributes to
        # 'SingleFilePSD' instance
        if self.is_rotor:
            myPSDs.N_blades = self.N_blades
            myPSDs.R_blades = self.R_blades

            if hasattr(self, 'rpm'):
                myPSDs.rpm = self.rpm
            if hasattr(self, 'f_shaft'):
                myPSDs.f_shaft = self.f_shaft
            if hasattr(self, 'bpf'):
                myPSDs.bpf = self.bpf
            if hasattr(self, 'Mtip'):
                myPSDs.Mtip = self.Mtip

        return myPSDs


    # *************************************************************************
    # Setter methods for RPM / shaft freq / BPF / blade tip Mach number

    def set_RPM(self, rpm):
        """
        Setter method for rotor RPM. Also writes shaft freq, BPF, and Mtip.
        """

        # Rotations per minute
        #   float
        self.rpm = rpm

        # shaft frequency [Hz]
        self.f_shaft = self.rpm/60

        # Blade passing frequency [Hz]
        self.bpf = self.f_shaft*self.N_blades

        # blade tip Mach number
        self.Mtip = self.rpm_to_Mtip(rpm)


    def set_fshaft(self, f_shaft):
        """
        Setter method for rotor shaft freq. Also writes RPM, BPF, and Mtip.
        """
        # shaft frequency [Hz]
        #   float
        self.f_shaft = f_shaft

        # Rotations per minute
        self.rpm = self.f_shaft*60

        # Blade passing frequency [Hz]
        self.bpf = self.f_shaft*self.N_blades

        # blade tip Mach number
        self.Mtip = self.rpm_to_Mtip(self.rpm)


    def rpm_to_Mtip(self, rpm, c0=340):
        """Converts RPM to blade tip Mach number"""

        v_tip = 2*np.pi*self.R_blades*(rpm/60)

        return v_tip/c0


    # *************************************************************************
    def test_flow_recirc(self, ch, bin_width=3, Ndft=DEFAULT_NDFT,
                         Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):
        """
        Calculates Overall, BPF, BPFx2, and BPFx3 levels over time from
        single-channel spectrogram data to assess flow recirculation effects.

        Parameters
        ----------
        ch : int
            Channel index to test for flow recirculation

        bin_width : int, optional
            Bandwidth of integration for BPF harmonics, in number of frequency
            bins. Default is 3 bins.

        Ndft : int, optional
            Length of DFT for creating the spectrogram, in samples. Default is
            2^13, and is set in 'sma_consts_aux.py' file.

        Ndft : int, optional
            Length of DFT segment for creating the spectrogram, in samples.
            Default is 2^13, and is set in 'sma_consts_aux.py' file.

        Noverlap : int, optional
            Length of segment overlap for creating the spectrogram, in samples.
            Default is 2^12 (Ndft/2), and is set in 'sma_consts_aux.py' file.

        window : str, optional
            Name of window type to use in spectrogram. This is passed to
            'scipy.signal.spectrogram', so any known Scipy window functions are
            accepted. Default is 'hann'.


        Returns
        -------
        time: (N_seg,)-shaped array_type
            Numpy array containing the initial times of each segment.

        levels: (4, N_seg)-shaped array_type
            Numpy array containing the integrated power of each signal metric
            (i.e. Overall, BPF, 2xBPF, 3xBPF) at each segment.

        names : list
            List of names of each signal metric.


        Notes
        -----
        This method works by showing possible changes in the recorded signal
        levels when a propulsor unit goes from off to a desired RPM. If flow
        recirculation is present, turbulent air in the device exhaust gets
        reingested at the inlet after some time, and this is evidenced by an
        increase in BPF harmonics' levels a few seconds after the unit is
        turned on. If no noticeable change in BPF levels is seen, it is assumed
        there is no flow recirculation under the conditions tested.

        The BPF harmonics' levels are calculated by finding the frequency bin
        closest to the actual BPF harmonic, and integrating the acoustic
        spectrum within +- 'bin_width'. Hence, if 'freq_bpf=10' and
        'bin_width=2', we sum the spectrum values within bins 8 and 12
        (inclusive).
        """

        assert hasattr(self, 'bpf'), \
            "Cannot assess flow recirculation effects - 'SingleFileRotorTime' instance has no attribute 'bpf'!"

        freq, time, Sxx = ss.spectrogram(self.mic_data[ch, :], self.fs,
                                         window=window, nperseg=Ndft,
                                         noverlap=Noverlap)
        df = freq[1]

        levels = np.zeros((4, time.shape[0]))

        # find frequency index for BPF
        freq_bpf = np.argmin(np.abs(freq - self.bpf))

        # overall level
        levels[0, :] = np.sum(Sxx, axis=0)*df

        # BPF and harmonics levels
        levels[1,:] = np.sum(Sxx[freq_bpf-bin_width : freq_bpf+bin_width+1, :], axis=0)*df
        levels[2,:] = np.sum(Sxx[2*freq_bpf-bin_width : 2*freq_bpf+bin_width+1, :], axis=0)*df
        levels[3,:] = np.sum(Sxx[3*freq_bpf-bin_width : 3*freq_bpf+bin_width+1, :], axis=0)*df

        names = ['Overall', '1xBPF', '2xBPF', '3xBPF']

        return time, levels, names


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

    
    # *************************************************************************
    def extract_rotor_angle(self, tacho_attrname):
        assert self.is_rotor is True,\
            "'SingleFileTimeSeries' instance attribute 'is_rotor' is false - can't perform synchronous averaging!"
        
        assert hasattr(self, tacho_attrname), \
            f"'SingleFileTimeSeries' instance does not have attribute '{tacho_attrname}'!"
        
        tacho = getattr(self, tacho_attrname)
    
        # extract rotor angle from tachometer signal
        f_low = 0.8*self.f_shaft
        f_high = 1.2*self.f_shaft
        filter_order = 3
        
        # instantaneous rotor angle in time domain
        self.angle_t = AR.extract_rotor_angle(tacho, self.fs, f_low, f_high,
                                              filter_order)
        
        return self.angle_t
        

    # *************************************************************************
    def synchr_averaging(self, tacho_attrname, N_interp, N_periods, max_Nt,
                         phase_shift=None):
        """
        Performs synchronous averaging of pressure signals over rotor angle
        domain. Uses 'N_interp' points per rotor revolution
        """
        
        angle_t = self.extract_rotor_angle(tacho_attrname)
        
        # apply phase shift to rotor angle (change theta=0 reference point)
        if phase_shift:
            angle_t = AR.shift_rotor_angle(angle_t, phase_shift)
    
        # Performs angular resampling of acoustic signal
        p_rlocked, angle_rlocked = AR.angular_resampling(self.mic_data[:, :max_Nt],
                                                         angle_t[:max_Nt],
                                                         N_interp)

        # estimate rotor-locked sampling freq
        T_shaft = 1/self.f_shaft
        fs_rlocked = N_interp/T_shaft
    
        # decompose rotor-locked signals into mean (averaged over 'N_periods')
        # and residue
        p_mean_rlocked, p_res_rlocked = AR.synchr_averaging(p_rlocked, N_interp, N_periods)
    
        return p_mean_rlocked, p_res_rlocked, angle_t, angle_rlocked, fs_rlocked


# #############################################################################
# %% Class 'SingleFilePSD'
# #############################################################################

class SingleFilePSD:
    """
    Class to store single-file, multichannel PSD and associated frequency-domain
    information. PSDs are assumed single-sided.
    """

    def __init__(self, filename, psd, freq, fs, Ndft=DEFAULT_NDFT,
                 Noverlap=DEFAULT_NOVERLAP, window=DEFAULT_WINDOW):

        # name of file where PSD data originates from
        self.filename = filename

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
            kernel_size = _round_to_nearest_odd(kernel_size_Hz/self.df)

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
            radius = _round_to_nearest_odd(radius_Hz/self.df)

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

