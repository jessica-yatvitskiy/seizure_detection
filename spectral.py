"""Tools for spectral analysis.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import fftpack
from scipy.signal import signaltools
from scipy.signal.windows import get_window
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
import warnings

from scipy._lib.six import string_types

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    scipy.signal._spectral_helper

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        func = fftpack.fft
    else:
        result = result.real
        func = np.fft.rfft
    result = func(result, n=nfft)

    return result


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        6
        window.
    """

    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, string_types) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg


def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See arXiv:gr-qc/0509116 Appendix B for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.
    """
    ii_2 = 2 * np.arange(1., (n-1) // 2 + 1)
    return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)

def compute_initial_coherence_state(x,window='hann',nfft=None,nperseg=None):
    """
    This function performs computation that is done in every call to _spectral_helper in scipy's implementation of coherence. This function needs to be called only once for array x of a given size

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    """
    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    initial_coherence_state={}
    initial_coherence_state["win"]=win
    initial_coherence_state["nperseg"]=nperseg
    initial_coherence_state["nfft"]=nfft
    return initial_coherence_state

def compute_coherence_state_for_one_side(x,initial_coherence_state,fs=1.0,detrend='constant',scaling='density',noverlap=None,boundary=None,mode='psd', axis=-1):
    """
    This function performs computation that is done two times in every call to coherence function in scipy's implementation of coherence. This function needs to be called only once per distinct array x 

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    """
    win=initial_coherence_state["win"]
    nperseg=initial_coherence_state["nperseg"] 
    nfft=initial_coherence_state["nfft"]

    x = np.asarray(x)
    outdtype = np.result_type(x, np.complex64)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    sides = 'onesided'
    freqs = np.fft.rfftfreq(nfft, 1/fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    if boundary is not None:
        time -= (nperseg/2) / fs

    same_side='True'
    P_self=compute_coherence(result,result,same_side,scale,mode,nfft,outdtype,axis)

    coherence_state={}
    coherence_state["result"]=result
    coherence_state["freqs"]=freqs
    coherence_state["time"]=time
    coherence_state["outdtype"]=outdtype
    coherence_state["scale"]=scale
    coherence_state["nfft"]=nfft
    coherence_state["mode"]=mode;
    coherence_state["axis"]=axis;
    coherence_state["P_self"]=P_self
    return coherence_state

def compute_coherence_from_states(x_state,y_state):
    """
    based on per-channel states returned by compute_coherence_state_for_one_side, compute coherence between X and Y

    Parameters
    ---------
    """
    result_x=x_state["result"]
    result_y=y_state["result"]
    Pxx=x_state["P_self"]
    Pyy=y_state["P_self"]
    freqs=x_state["freqs"]
    time=x_state["time"]
    scale=x_state["scale"]
    outdtype=x_state["outdtype"]
    nfft=x_state["nfft"]
    mode=x_state["mode"]
    axis=x_state["axis"]

    same_side='False'
    Pxy=compute_coherence(result_x,result_y,same_side,scale,mode,nfft,outdtype,axis)
    Cxy = np.abs(Pxy)**2 / Pxx.real / Pyy.real

    return freqs, Cxy

def compute_coherence(result_x,result_y,same_side,scale,mode,nfft,outdtype,axis,average='mean'):
    #this is the only part of Scipy's implementation of coherence that actually has to be executed for each pair of distinct X1, X2, X3, ... XN arrays
    if same_side==True:
        result = np.conjugate(result_x) * result_x
    else:
        result = np.conjugate(result_x) * result_y
    result *= scale

    if mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    result = result.astype(outdtype)

    if same_side==True:
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.rollaxis(result, -1, axis)
    if same_side==True:
        result = result.real

    # Average over windows.
    if len(result.shape) >= 2 and result.size > 0:
        if result.shape[-1] > 1:
            if average == 'median':
                result = np.median(result, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == 'mean':
                result = result.mean(axis=-1)
            else:
                raise ValueError('average must be "median" or "mean", got %s'
                                 % (average,))
        else:
            result = np.reshape(result, result.shape[:-1])
    return result
