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


# #############################################################################
# %% Functions to process rotor angle from tacho signal
# #############################################################################

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


# #############################################################################
def shift_rotor_angle(rotor_angle, phase_shift, angle_units='deg'):
    """
    Applies a phase shift to a rotor angle variable. Input 'rotor_angle' should
    be between 0 and 2*pi.

    Parameters
    ----------
    rotor_angle : (Nt,)-shape array_like
        Numpy array containing instantaneous rotor angle
    
    phase_shift : float
        Amount of phase shift, in degrees (default) or radians.

    angle_units : {'deg', 'rad'}, optional.
        Flag determining whether phase shift is in degrees or radians. Default
        is degrees.
    
    Returns
    -------
    rotor_angle_shifted : (Nt,)-shape array_like
        Instantaneous rotor angle shifted by 'phase_shift'.
    """
    
    if angle_units == 'deg':
        phase_shift *= np.pi/180
    
    _, rotor_angle_shifted = np.divmod(rotor_angle + phase_shift, 2*np.pi)
    
    return rotor_angle_shifted
    

# #############################################################################
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


# #############################################################################
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


# #############################################################################
def angular_resampling(signals, rotor_angle, N_interp, ignore_last_cycle=True):
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
    
    ignore_last_cycle : boolean, optional
        Flag to determine whether to ignore last full cycle, as it might be 
        corrupted. Default is True

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
    # onto rotor angle 'theta'.
    
    if ignore_last_cycle:
        # Last full cycle may be corrupted
        N_cycles -= 1
        
    angular_signals = np.zeros((N_ch, N_cycles*N_interp))
    angle_signal = np.zeros(N_cycles*N_interp)

    # for each full cycle in recording (except the last)...
    for n_c in range(N_cycles):

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


# #############################################################################
def synchr_averaging(signals, N_per_cycle, N_periods):
    """
    Performs synchronous averaging of an input 'signals' containing 
    multichannel, resampled signals with exactly 'N_per_cycle' samples per
    period. Outputs are the mean, synchronously-averaged signal calculated over
    one full period, and the residue signal (after subtracting the mean).
    
    Parameters
    ----------
    signals : (N_ch, N_t)-shape array_like
        Numpy array containing 'N_ch' signals with exactly 'N_cycles' full
        cycles (so N_t = N_cycles*N_per_cycle)

    N_per_cycle : int
        Number of samples per cycle
    
    N_periods : int
        Number of integer periods to include in output signal.

    Returns
    -------
    mean_signal : (N_ch, N_periods*N_per_cycle)-shape array_like
        Numpy array containing 'N_ch' mean signals per 'N_periods'

    residue_signal : (N_ch, Nt)-shape array_like
        Numpy array containing 'N_ch' residue signals
    """
    
    N_ch, N_t = signals.shape
    
    # zero-pad signals if necessary
    N_zero_pad = N_t%(N_periods*N_per_cycle)
    if N_zero_pad != 0:
        zero_padding = np.zeros((N_ch, N_zero_pad))
        signals = np.concatenate((signals, zero_padding), axis=1)
    
    signals_period = signals.reshape((N_ch, N_periods*N_per_cycle, -1),
                                     order='F')
    
    mean = signals_period.mean(axis=2)
    
    residue = (signals_period - mean[:, :, np.newaxis]).reshape((N_ch, -1), order='F')
    
    return mean, residue
    