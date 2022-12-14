"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono

Test script for single-file classes: reads single file containing time-domain
data, calculate spectra, calculate some SPL metrics.

Author:
    Fabio Casagrande Hirono
    November 2022
"""

import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

import salford_mic_arc as SMA

# ref acoustic pressure (20 uPa RMS)
p_ref = SMA.P_REF

save_fig = False

# %% Create instance of 'InputFile'

# filename
filename = 'data/2022_09_12/IPM5_phi075_6000rpm_180deg_2022_09_12.h5'

# # optional: print HDF5 file structure
# SMA.print_hdf5_file_structure(filename)

# create list of microphone channel names
mic_chs = ['Mic_00deg',
           'Mic_10deg',
           'Mic_20deg',
           'Mic_30deg',
           'Mic_40deg',
           'Mic_50deg',
           'Mic_60deg',
           'Mic_70deg',
           'Mic_80deg',
           'Mic_90deg']

# create list of other channels to read
other_chs = ['RPM', 'RevCounter', 'LoadCell1']

# recording length [s]
T = 30

# sampling freqs [Hz]
fs = 50000
fs2 = 12500

# number of rotor blades
N_blades = 8

# rotor blade radius
R_blades = 0.145

# name of attribute containing RPM value
rpm_name = 'mean_RPM'

# create instance of InputFile
ipm_inputfile = SMA.InputFile(filename)

# set file metadata
ipm_inputfile.set_mic_channel_names(mic_chs)
ipm_inputfile.set_other_ch_names(other_chs)
ipm_inputfile.set_recording_length(T)
ipm_inputfile.set_sampling_freq(fs)
ipm_inputfile.set_sampling_freq2(fs2)

# set rotor metadata
ipm_inputfile.is_rotor = True
ipm_inputfile.set_N_blades(N_blades)
ipm_inputfile.set_R_blades(R_blades)
ipm_inputfile.set_rpm_attr_name(rpm_name)

# print instance attributes by viewing it as a dict
print(ipm_inputfile.__dict__.keys())

# %% read raw data from Dewesoft HDF5 file using 'SingleFileTimeSeries' class

ipm_data = SMA.SingleFileTimeSeries(ipm_inputfile)

# calculate mean value of channels listed in 'other_chs'
print("Mean thrust : {:.2f} N".format(ipm_data.mean_LoadCell1))
print("Mean RPM : {:.1f} ".format(ipm_data.mean_RPM))


# %% optional tasks: filter data, estimate peak frequency location, export mic
# signals as .WAV files


# # filter mic data using 3rd order highpass Butterworth filter at 50 Hz cutoff
# ipm_data.filter_data(filter_order=3, fc=50, btype='highpass')

# estimate blade-passing frequency by finding peak between 750 and 850 Hz
# --> f_low, f_high are determined by visual inspection of PSDs!
f_bpf = ipm_data.estimate_peak_freq(f_low=750, f_high=850, Ndft=2**14)

# # export mic signals as multichannel .WAV files - see function documentation
# wav_filename = 'IPM5_8ch_testfile'
# ipm_data.export_wavs(wav_filename, channels=8)

# %% Create 'SingleFilePSD' instance from raw time series data

# define frequency range of interest, in Hz
f_low = 150
f_high = 10000

ipm_PSD = ipm_data.calc_PSDs(Ndft=2**13, Noverlap=2**12, window='hann')


# calculate broadband component of PSD
ipm_PSD.calc_broadband_PSD()

# find peaks in PSD in interval [f_low, f_high]
ipm_PSD.find_peaks(f_low, f_high)

# *****************************************************************************
# calculate SPLs by integrating the respective PSDs

# obtain overall SPL (broadband + peaks) within range [f_low, f_high]
#overall_SPL = ipm_PSD.calc_overall_SPL(f_low, f_high)
ipm_PSD.calc_overall_SPL(f_low, f_high)

# obtain broadband SPL within range [f_low, f_high]
# broadband_SPL = ipm_PSD.calc_broadband_SPL(f_low, f_high)
ipm_PSD.calc_broadband_SPL(f_low, f_high)

# obtain tonal SPL (sum of tones' SPL - BPF harmonics or otherwise)
# tonal_SPL = ipm_PSD.calc_tonal_SPL()
ipm_PSD.calc_tonal_SPL()


# %% select one channel to plot PSD and print SPLs

ch = 0
peak_indices = ipm_PSD.peak_indices[ch, :]

# plot PSD, highlighting broadband component and tonal peaks
plt.figure()

plt.semilogx(ipm_PSD.freq[1:],
             10*np.log10(ipm_PSD.psd[ch, 1:]/(SMA.P_REF**2)),
             'k:', label='PSD')

plt.semilogx(ipm_PSD.freq[1:],
             10*np.log10(ipm_PSD.psd_broadband[ch, 1:]/(SMA.P_REF**2)),
             'r-', label='Broadband')

plt.semilogx(ipm_PSD.freq[peak_indices],
             10*np.log10(ipm_PSD.psd[ch, peak_indices]/(SMA.P_REF**2)),
             'g^', label='Peaks')
plt.legend()
plt.grid()
plt.xlim([100, 20e3])
plt.ylim([0, 50])

plt.title('Channel {} PSD'.format(ch))
plt.ylabel('Magnitude [dB ref (20 uPa)^2/Hz]')
plt.xlabel('Frequency [Hz]')

# print SPLs for chosen channel
print('Channel {}:'.format(ch))
print('\tOverall SPL    = {:.1f} dB'.format(ipm_PSD.overall_SPL[ch]))
print('\tBroadband SPL  = {:.1f} dB'.format(ipm_PSD.broadband_SPL[ch]))
print('\tTonal SPL      = {:.1f} dB'.format(ipm_PSD.tonal_SPL[ch]))
