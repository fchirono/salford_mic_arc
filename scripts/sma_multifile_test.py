"""
salford_mic_arc - a Python package for reading directivity measurement data
https://github.com/fchirono/salford_mic_arc
Copyright (c) 2022, Fabio Casagrande Hirono


Test script for multiple-file classes: reads multiple files made from the same
unit under test at different azimuthal conditions, calculates spectra and SPLs,
plot directivities for given SPL metrics.

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

# %% read and inspect file

# list of filenames
ipm_filenames = ['data/2022_09_12/IPM5_phi075_6000rpm_000deg_2022_09_12.h5',
                 'data/2022_09_12/IPM5_phi075_6000rpm_180deg_2022_09_12.h5',]


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


# read raw time series from Dewesoft HDF5 file
ipm_data = SMA.MultiFileTimeSeries(ipm_filenames, mic_chs, T=30, fs=50000,
                                    other_ch_names=other_chs, fs2=12500)


# calculate mean value of channels listed in 'other_chs'
ipm_data.calc_channel_mean(other_chs)
print('Mean RPM: {}'.format(ipm_data.mean_RPM))
print('Mean thrust [N]: {}'.format(ipm_data.mean_LoadCell1))


# # filter mic data using 3rd order highpass Butterworth filter at 50 Hz cutoff
# ipm_data.filter_data(filter_order=3, fc=50, btype='highpass')


# %% Create 'MultiChannelPSD' instance from raw time series data

# define frequency range of interest, in Hz
f_low = 150
f_high = 10000

ipm_PSD = SMA.MultiFilePSD(ipm_filenames, mic_chs)

# *****************************************************************************
# calculate SPLs by integrating the respective PSDs

# obtain overall SPL (broadband + peaks) within range [f_low, f_high]
#overall_SPL = ipm_PSD.calc_overall_SPL(f_low, f_high)
ipm_PSD.calc_overall_SPL(f_low, f_high)

# obtain broadband SPL within range [f_low, f_high]
# broadband_SPL = ipm_PSD.calc_broadband_SPL(f_low, f_high)
ipm_PSD.calc_broadband_SPL(f_low, f_high)

# find peaks in PSD in interval [f_low, f_high]
ipm_PSD.find_peaks(f_low, f_high)

# obtain tonal SPL (sum of tones' SPL - BPF harmonics or otherwise)
# tonal_SPL = ipm_PSD.calc_tonal_SPL()
ipm_PSD.calc_tonal_SPL()



# %% outputs SPL calculations to polar format

polar_angles, overall_SPL_polar = ipm_PSD.SPL_to_polar('overall_SPL',
                                                       combine_90deg=False)

_, broadband_SPL_polar = ipm_PSD.SPL_to_polar('broadband_SPL',
                                              combine_90deg=False)

_, tonal_SPL_polar = ipm_PSD.SPL_to_polar('tonal_SPL',
                                          combine_90deg=False)

SPL_min = 30
SPL_max = 80

fig = plt.figure(figsize=(8,5.5))
ax = fig.add_subplot(projection='polar')

ax.plot(polar_angles*np.pi/180, overall_SPL_polar, linewidth=3,
        label='Overall SPL')
ax.plot(polar_angles*np.pi/180, broadband_SPL_polar, linewidth=3,
        linestyle='--', label='Broadband SPL')
ax.plot(polar_angles*np.pi/180, tonal_SPL_polar, linewidth=3,
        linestyle='-.', label='Tonal SPL')

ax.legend(loc=(0.825, 0.65))
ax.set_ylim([SPL_min, SPL_max])
ax.set_xlabel('SPL [dB re 20 uPa]', fontsize=15, labelpad=-80)

# set angle sector to half circle
ax.set_thetamin(0)
ax.set_thetamax(180)
ax.set_title('SPLs', y=0.82, fontsize=20)

# workaround on axes size to remove empty space around plot
ax.set_position( [0.05, -0.5, 0.85, 2])
