import pytest
import pynapple as nap
import numpy as np
from scipy import signal


@pytest.mark.parametrize("freq", [10, 100])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
        nap.IntervalSet(start=[0, 0.5, 0.95], end=[0.4, 0.9, 1])
    ]
)
def test_filtering_single_freq(freq, order, btype, shape: tuple, ep):

    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1]*(len(shape) - 1)) + np.random.normal(size=shape))

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    out = nap.compute_filtered_signal(tsd, freq_band=freq, filter_type=btype, order=order)
    b, a = signal.butter(order, freq, fs=tsd.rate, btype=btype)
    out_sci = []
    for iset in ep:
        out_sci.append(signal.filtfilt(b, a, tsd.restrict(iset).d, axis=0))
    out_sci = np.concatenate(out_sci, axis=0)
    np.testing.assert_array_equal(out.d, out_sci)
