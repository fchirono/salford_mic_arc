# -*- coding: utf-8 -*-
"""
Functions to implement angular resampling of noise from rotating machines. Uses
a tachometer signal (e.g. 1/rev pulse) as reference for the rotor position, and
resamples the acoustic signals with an integer number of samples per rotor 
rotation.

Recommended import syntax:

>>> from salford_mic_arc import angular_resampling


Author:
    Fabio Casagrande Hirono
    Feb 2023
"""

import numpy as np
import scipy.signal as ss

import scipy.interpolate as si


# **********************************************************************
# %% Functions to process rotor angle from tacho signal
# **********************************************************************

def extract_rotor_angle(tacho_data, fs, f_low, f_high, filter_order):
    """
    Extracts the rotor angular position (between 0 and 2*pi rad) from the
    tachometer (1/rev pulses) signal, and returns the angular position as a
    positive-ramp sawtooth time series with amplitude between 0 and 2*pi rad.

    The tacho signal is bandpassed around its fundamental frequency using
    forward-backwards filtering with a Butterworth filter, where 'f_low',
    'f_high', and 'filter_order' are the filter parameters. The phase of
    this narrowband signal (i.e. the rotor angular position) is estimated
    using the Hilbert Transform.

    Parameters
    ----------
    tacho_data : (N_t,)-shape array_like
        Numpy array containing the tachometer signal (1/rev pulses)

    fs : float
        Sampling frequency of tachometer signal, in Hz

    f_low : float
        Low-frequency filter cutoff

    f_high : float
        High-frequency filter cutoff

    filter_order : int
        Filter order

    Returns
    -------
    angle : (N_t,)-shape array_like
        Numpy array containing the rotor angular position time series,
        between 0 and 2*pi rad.
    """

    # create bandpass filter to extract fundamental frequency
    bandpass_filter = ss.butter(filter_order, [f_low, f_high], 'bandpass',
                                output='sos', fs=fs)

    # fwd-bkwd filtering of the tacho signal
    bp_tacho_data = ss.sosfiltfilt(bandpass_filter, tacho_data)


    # obtain analytic signal (real + imag) using Hilbert Transform
    analytic_tacho = ss.hilbert(bp_tacho_data)

    # rotor angle is the instantaneous phase of analytic signal,
    # between 0 and 2*pi rad
    rotor_angle = np.angle(analytic_tacho) + np.pi

    return rotor_angle


def extract_rotor_zeros(rotor_angle):
    """
    Returns a list of indices of where the rotor angular angle time series
    reaches 0 deg.

    This function explores the fact that the periodic transitions between
    2*pi and 0 trigger large peaks in the derivative of 'rotor_angle'.

    Parameters
    ----------
    rotor_angle : (N_t,)-shape array_like
        Numpy array containing the sawtooth-like rotor angle time series

    Returns
    -------
    phase_zeros : list
        List containing the indices of where 'rotor_angle' reaches zero.
    """

    phase_diff = np.diff(rotor_angle)
    phase_zeros = ss.find_peaks(np.abs(phase_diff), height=1)[0] +1

    return phase_zeros


def extract_rotor_freq(rotor_angle, fs):
    """
    Estimates rotor instantaneous rotational frequency, in Hz, from rotor angle
    signal.

    Rotor frequency at each time step is estimated using the derivative of the
    unwrapped rotor angle signal (i.e. the phase of the analytical signal).

    Parameters
    ----------
    rotor_angle : (N_t,)-shape array_like
        Numpy array containing the sawtooth-like rotor angle time series

    fs : float
        Sampling frequency of rotor angle signal, in Hz

    Returns
    -------
    inst_freq : (N_t,)-shape array_like
        Numpy array containing the estimated rotor frequency, in Hz.
    """

    inst_freq = (np.diff(np.unwrap(rotor_angle))/(2*np.pi))*fs

    return inst_freq


def angular_resampling(signals, rotor_angle, N_interp):
    """
    Resamples time-domain signals to rotor angle domain, so output signal
    samples are locked with rotor angular position and have 'N_interp' samples
    per rotor cycle exactly.

    Returns 'angular_signals' and 'angle_signal', containing 'N_cycles'
    equally-sampled full rotor cycles.

    Parameters
    ----------
    signals : (N_ch, N_t)-shape array_like
        Numpy array containing all 'N_ch' signals in time-domain to be
        resampled.

    rotor_angle : (N_t,)-shape array_like
        Numpy array containing the rotor angle signal, between 0 and 2*pi rads

    N_interp : int
        Number of samples to use in interpolation (per cycle)

    Returns
    -------
    angular_signals : (N_ch, N_cycles*N_interp)-shape array_like
        Numpy array containing all 'N_ch' signals resampled to angular domain

    angle_signal : (N_cycles*N_interp,)-shape array_like
        Numpy array containing resampled angle time series
    """

    # No. channels, No. of time samples
    N_ch, N_t = signals.shape

    # angular sampling interval (rad/sample)
    dtheta = 2*np.pi/N_interp

    # rotor angle (uniformly sampled)
    theta = np.linspace(0, 2*np.pi - dtheta, N_interp)

    # find rotor angle zero crossings
    rotor_zeros = extract_rotor_zeros(rotor_angle)
    N_zeros = rotor_zeros.shape[0]

    # Number of integer cycles present in angular signals
    N_cycles = N_zeros-1

    # **********************************************************************
    # interpolate each cycle (plus safety buffer of 10 samples on each side)
    # onto rotor angle 'theta'. Ignore last full cycle, which can be corrupted

    angular_signals = np.zeros((N_ch, (N_cycles-1)*N_interp))
    angle_signal = np.zeros((N_cycles-1)*N_interp)

    # for each full cycle in recording (except the last)...
    for n_c in range(N_cycles-1):

        # find indices a bit before crossing and a bit after next crossing
        n_start = rotor_zeros[n_c] - 10
        n_end = rotor_zeros[n_c+1] + 10

        # copy and unwrap angle values, so they start slightly below 0 and end
        # slightly above 2*pi
        angle_segment = np.unwrap(rotor_angle[n_start : n_end]) - 2*np.pi

        # interpolate signals over angle segment
        interpolator = si.interp1d(angle_segment, signals[:, n_start:n_end],
                                   axis=-1, kind='cubic')

        angular_signals[:, n_c*N_interp : (n_c+1)*N_interp] = interpolator(theta)
        angle_signal[n_c*N_interp : (n_c+1)*N_interp] = theta

    # **********************************************************************

    return angular_signals, angle_signal

