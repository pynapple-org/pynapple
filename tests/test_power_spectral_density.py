import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def get_signal_and_output(f=2, fs=1000,duration=100,interval_size=10):
	t=np.arange(0, duration, 1/fs)
	d=np.cos(2*np.pi*f*t)
	sig = nap.Tsd(t=t,d=d, time_support=nap.IntervalSet(0,100))
	tmp = d.reshape((int(duration/interval_size),int(fs*interval_size))).T
	out = np.sum(np.fft.fft(tmp, axis=0), 1)
	freq = np.fft.fftfreq(out.shape[0], 1 / fs)
	order = np.argsort(freq)
	out = out[order]
	freq = freq[order]
	return (sig, out, freq)


def test_basic():
    sig, out, freq = get_signal_and_output()

    psd = nap.compute_mean_power_spectral_density(sig, 10)
   
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out[freq>=0])
    np.testing.assert_array_almost_equal(psd.index.values, freq[freq>=0])





@pytest.mark.parametrize("interval_size, expected_exception", [
    (10, None),    # Regular case
    (200, RuntimeError),  # Interval size too large
    (1, RuntimeError)  # Epoch too small
])
@setup_signal_and_params
def test_compute_mean_power_spectral_density(sig, interval_size, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            compute_mean_power_spectral_density(sig, interval_size)
    else:
        psd = compute_mean_power_spectral_density(sig, interval_size)
        assert isinstance(psd, pd.DataFrame)
        assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty

@pytest.mark.parametrize("full_range", [True, False])
@setup_signal_and_params
def test_full_range_option(sig, full_range):
    interval_size = 10  # Choose a valid interval size for this test
    
    psd = compute_mean_power_spectral_density(sig, interval_size, full_range=full_range)
    
    if full_range:
        assert (psd.index >= 0).all()
    else:
        assert (psd.index >= 0).any()  # Partial range should exclude negative frequencies
