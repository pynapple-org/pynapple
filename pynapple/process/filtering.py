"""
	Filtering module
"""

import numpy as np
from .. import core as nap
from scipy.signal import butter, lfilter, filtfilt


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_bandpass_filter(data, freq_band, sampling_frequency=None, order=4):
    """
    Bandpass filtering the LFP.
    
    Parameters
    ----------
    data : Tsd/TsdFrame
        Description
    lowcut : TYPE
        Description
    highcut : TYPE
        Description
    fs : TYPE
        Description
    order : int, optional
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    time_support = data.time_support
    time_index = data.as_units('s').index.values
    if type(data) is nap.TsdFrame:
        tmp = np.zeros(data.shape)
        for i in np.arange(data.shape[1]):
            tmp[:,i] = bandpass_filter(data[:,i], lowcut, highcut, fs, order)

        return nap.TsdFrame(
            t = time_index,
            d = tmp,
            time_support = time_support,
            time_units = 's',
            columns = data.columns)

    elif type(data) is nap.Tsd:
        flfp = _butter_bandpass_filter(data.values, lowcut, highcut, fs, order)
        return nap.Tsd(
            t=time_index,
            d=flfp,
            time_support=time_support,
            time_units='s')

    else:
        raise RuntimeError("Unknow format. Should be Tsd/TsdFrame")