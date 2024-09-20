"""Tests of tuning curves for `pynapple` package."""
from contextlib import nullcontext as does_not_raise
import pynapple as nap
import numpy as np
import pandas as pd
import pytest

########################
# Type Error
########################
def get_group():
    return nap.TsGroup({0: nap.Ts(t=np.arange(0, 100))})
def get_feature():
    return nap.Tsd(
        t=np.arange(0, 100, 0.1), d=np.arange(0, 100, 0.1) % 1.0,
        time_support = nap.IntervalSet(0, 100)
        )
def get_features():
    tmp = np.vstack(
        (np.repeat(np.arange(0, 100), 10), np.tile(np.arange(0, 100), 10))
    ).T
    return nap.TsdFrame(
        t=np.arange(0, 200, 0.1), d=np.vstack((tmp, tmp[::-1])),
        time_support = nap.IntervalSet(0, 200)
        )
def get_ep():
    return nap.IntervalSet(start=0, end=50)
def get_tsdframe():
    return nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 2)))

@pytest.mark.parametrize("group, dict_ep, expected_exception", [
    ("a", {0:nap.IntervalSet(start=0, end=50),1:nap.IntervalSet(start=50, end=100)}, pytest.raises(TypeError,match="group should be a TsGroup.")),
    (get_group(), "a", pytest.raises(TypeError,match="dict_ep should be a dictionary of IntervalSet")),
    (get_group(), {0:"a",1:nap.IntervalSet(start=50, end=100)}, pytest.raises(TypeError,match="dict_ep argument should contain only IntervalSet.")),
])
def test_compute_discrete_tuning_curves_errors(group, dict_ep, expected_exception):
    with expected_exception:
        nap.compute_discrete_tuning_curves(group, dict_ep)

@pytest.mark.parametrize("group, feature, nb_bins, ep, minmax, expected_exception", [
    ("a",get_feature(), 10, get_ep(), (0, 1), "group should be a TsGroup."),
    (get_group(),"a", 10, get_ep(), (0, 1), r"feature should be a Tsd \(or TsdFrame with 1 column only\)"),
    (get_group(),get_feature(), "a", get_ep(), (0, 1), r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\)."),
    (get_group(),get_feature(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
    (get_group(),get_feature(), 10, get_ep(), 1, r"minmax should be a tuple\/list of 2 numbers"),
])
def test_compute_1d_tuning_curves_errors(group, feature, nb_bins, ep, minmax, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_1d_tuning_curves(group, feature, nb_bins, ep, minmax)

@pytest.mark.parametrize("group, features, nb_bins, ep, minmax, expected_exception", [
    ("a",get_features(), 10, get_ep(), (0, 1), "group should be a TsGroup."),
    (get_group(),"a", 10, get_ep(), (0, 1), r"features should be a TsdFrame with 2 columns"),
    (get_group(),get_features(), "a", get_ep(), (0, 1), r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\)."),
    (get_group(),get_features(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
    (get_group(),get_features(), 10, get_ep(), 1, r"minmax should be a tuple\/list of 2 numbers"),
])
def test_compute_2d_tuning_curves_errors(group, features, nb_bins, ep, minmax, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_2d_tuning_curves(group, features, nb_bins, ep, minmax)

@pytest.mark.parametrize("tc, feature, ep, minmax, bitssec, expected_exception", [
    ("a", get_feature(), get_ep(), (0, 1), True, "Argument tc should be of type pandas.DataFrame or numpy.ndarray"),
    (pd.DataFrame(),"a", get_ep(), (0, 1), True, r"feature should be a Tsd \(or TsdFrame with 1 column only\)"),
    (pd.DataFrame(), get_feature(), "a", (0, 1), True, r"ep should be an IntervalSet"),
    (pd.DataFrame(), get_feature(), get_ep(), 1, True, r"minmax should be a tuple\/list of 2 numbers"),
    (pd.DataFrame(), get_feature(), get_ep(), (0,1), "a", r"Argument bitssec should be of type bool"),
])
def test_compute_1d_mutual_info_errors(tc, feature, ep, minmax, bitssec, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_1d_mutual_info(tc, feature, ep, minmax, bitssec)

@pytest.mark.parametrize("dict_tc, features, ep, minmax, bitssec, expected_exception", [
    ("a", get_features(), get_ep(), (0, 1), True, "Argument dict_tc should be a dictionary of numpy.ndarray"),
    ({0:np.zeros((2,2))},"a", get_ep(), (0, 1), True, r"features should be a TsdFrame with 2 columns"),
    ({0:np.zeros((2,2))}, get_features(), "a", (0, 1), True, r"ep should be an IntervalSet"),
    ({0:np.zeros((2,2))}, get_features(), get_ep(), 1, True, r"minmax should be a tuple\/list of 2 numbers"),
    ({0:np.zeros((2,2))}, get_features(), get_ep(), (0,1), "a", r"Argument bitssec should be of type bool"),
])
def test_compute_2d_mutual_info_errors(dict_tc, features, ep, minmax, bitssec, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_2d_mutual_info(dict_tc, features, ep, minmax, bitssec)

@pytest.mark.parametrize("tsdframe, feature, nb_bins, ep, minmax, expected_exception", [
    ("a",get_feature(), 10, get_ep(), (0, 1), "Argument tsdframe should be of type Tsd or TsdFrame."),
    (get_tsdframe(),"a", 10, get_ep(), (0, 1), r"feature should be a Tsd \(or TsdFrame with 1 column only\)"),
    (get_tsdframe(),get_feature(), "a", get_ep(), (0, 1), r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\)."),
    (get_tsdframe(),get_feature(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
    (get_tsdframe(),get_feature(), 10, get_ep(), 1, r"minmax should be a tuple\/list of 2 numbers"),
])
def test_compute_1d_tuning_curves_continuous_errors(tsdframe, feature, nb_bins, ep, minmax, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_1d_tuning_curves_continuous(tsdframe, feature, nb_bins, ep, minmax)

@pytest.mark.parametrize("tsdframe, features, nb_bins, ep, minmax, expected_exception", [
    ("a",get_features(), 10, get_ep(), (0, 1), "Argument tsdframe should be of type Tsd or TsdFrame."),
    (get_tsdframe(),"a", 10, get_ep(), (0, 1), r"features should be a TsdFrame with 2 columns"),
    (get_tsdframe(),get_features(), "a", get_ep(), (0, 1), r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\)."),
    (get_tsdframe(),get_features(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
    (get_tsdframe(),get_features(), 10, get_ep(), 1, r"minmax should be a tuple\/list of 2 numbers"),
])
def test_compute_2d_tuning_curves_continuous_errors(tsdframe, features, nb_bins, ep, minmax, expected_exception):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_2d_tuning_curves_continuous(tsdframe, features, nb_bins, ep, minmax)

########################
# ValueError test
########################
@pytest.mark.parametrize("func, args, minmax, expected", [
    (nap.compute_1d_tuning_curves, (get_group(), get_feature(), 10), (0,1,2), "minmax should be of length 2."),
    (nap.compute_2d_tuning_curves, (get_group(), get_features(), 10), (0,1,2), "minmax should be of length 4."),
    (nap.compute_2d_tuning_curves, (get_group(), nap.TsdFrame(t=np.arange(10),d=np.ones((10,3))), 10), (0,1), "features should have 2 columns only."),
    (nap.compute_1d_tuning_curves, (get_group(), nap.TsdFrame(t=np.arange(10),d=np.ones((10,3))), 10), (0,1), r"feature should be a Tsd \(or TsdFrame with 1 column only\)"),
])
def test_compute_tuning_curves_value_error(func, args, minmax, expected):
    with pytest.raises(ValueError, match=expected):
        func(*args, minmax=minmax)


########################
# Normal test
########################
@pytest.mark.parametrize("group", [
    get_group()
])
@pytest.mark.parametrize("dict_ep", [
    { 0:nap.IntervalSet(start=0, end=50), 1:nap.IntervalSet(start=50, end=100)},
    { "0":nap.IntervalSet(start=0, end=50), "1":nap.IntervalSet(start=50, end=100)}
])
def test_compute_discrete_tuning_curves(group, dict_ep):
    tc = nap.compute_discrete_tuning_curves(group, dict_ep)
    assert len(tc) == 2
    assert list(tc.columns) == list(group.keys())
    assert list(tc.index.values) == list(dict_ep.keys())
    np.testing.assert_almost_equal(tc.iloc[0,0], 51/50)
    np.testing.assert_almost_equal(tc.iloc[1,0], 1)

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("args, kwargs, expected", [
    ((get_group(), get_feature(), 10), {}, np.array([10.0]+[0.0]*9)[:,None]),
    ((get_group(), get_feature(), 10), {"ep":get_ep()}, np.array([10.0]+[0.0]*9)[:,None]),
    ((get_group(), get_feature(), 10), {"minmax":(0, 0.9)}, np.array([10.0]+[0.0]*9)[:,None]),
    ((get_group(), get_feature(), 20), {"minmax":(0, 1.9)}, np.array([10.0]+[0.0]*9+[np.nan]*10)[:,None])
])
def test_compute_1d_tuning_curves(args, kwargs, expected):
    tc = nap.compute_1d_tuning_curves(*args, **kwargs)
    # Columns
    assert list(tc.columns) == list(args[0].keys())

    # Index
    assert len(tc) == args[2]
    if "minmax" in kwargs:
        tmp = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], args[2] + 1)
    else:
        tmp = np.linspace(np.min(args[1]), np.max(args[1]), args[2] + 1)
    np.testing.assert_almost_equal(tmp[0:-1] + np.diff(tmp)/2, tc.index.values)

    # Array
    np.testing.assert_almost_equal(tc.values, expected)

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("args, kwargs, expected", [
    ((get_group(), get_features(), 10), {}, np.ones((10,10))*0.5),
    ((get_group(), get_features(), 10), {"ep":nap.IntervalSet(0, 400)}, np.ones((10,10))*0.25),
    ((get_group(), get_features(), 10), {"minmax":(0, 100, 0, 100)}, np.ones((10,10))*0.5),
    ((get_group(), get_features(), 10), {"minmax":(0, 200, 0, 100)}, np.vstack((np.ones((5,10))*0.5,np.ones((5,10))*np.nan))),
])
def test_compute_2d_tuning_curves(args, kwargs, expected):
    tc, xy = nap.compute_2d_tuning_curves(*args, **kwargs)
    assert isinstance(tc, dict)

    # Keys
    assert list(tc.keys()) == list(args[0].keys())

    # Index
    assert isinstance(xy, list)
    assert len(xy) == 2
    if "minmax" in kwargs:
        tmp1 = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], args[2] + 1)
        tmp2 = np.linspace(kwargs["minmax"][2], kwargs["minmax"][3], args[2] + 1)
    else:
        tmp1 = np.linspace(np.min(args[1][:,0]), np.max(args[1][:,0]), args[2] + 1)
        tmp2 = np.linspace(np.min(args[1][:,1]), np.max(args[1][:,1]), args[2] + 1)

    np.testing.assert_almost_equal(tmp1[0:-1] + np.diff(tmp1)/2, xy[0])
    np.testing.assert_almost_equal(tmp2[0:-1] + np.diff(tmp2)/2, xy[1])

    # Values
    for i in tc.keys():
        assert tc[i].shape == (args[2], args[2])
        np.testing.assert_almost_equal(tc[i], expected)







def test_compute_1d_mutual_info():
    tc = pd.DataFrame(index=np.arange(0, 2), data=np.array([0, 10]))
    feature = nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50))
    si = nap.compute_1d_mutual_info(tc, feature)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ["SI"]
    assert list(si.index.values) == list(tc.columns)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 1.0)
    si = nap.compute_1d_mutual_info(tc, feature, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 5.0)

    ep = nap.IntervalSet(start=0, end=49)
    si = nap.compute_1d_mutual_info(tc, feature, ep=ep)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 1.0)
    si = nap.compute_1d_mutual_info(tc, feature, ep=ep, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 5.0)

    minmax = (0, 1)
    si = nap.compute_1d_mutual_info(tc, feature, minmax=minmax)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 1.0)

def test_compute_1d_mutual_info_array():    
    tc = np.array([[0],[10]])
    feature = nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50))
    si = nap.compute_1d_mutual_info(tc, feature)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ["SI"]    
    np.testing.assert_approx_equal(si.loc[0, "SI"], 1.0)
    si = nap.compute_1d_mutual_info(tc, feature, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 5.0)

def test_compute_2d_mutual_info():
    dict_tc = {0: np.array([[0, 1], [0, 0]])}
    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    si = nap.compute_2d_mutual_info(dict_tc, features)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ["SI"]
    assert list(si.index.values) == list(dict_tc.keys())
    np.testing.assert_approx_equal(si.loc[0, "SI"], 2.0)
    si = nap.compute_2d_mutual_info(dict_tc, features, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 0.5)

    ep = nap.IntervalSet(start=0, end=7)
    si = nap.compute_2d_mutual_info(dict_tc, features, ep=ep)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 2.0)
    si = nap.compute_2d_mutual_info(dict_tc, features, ep=ep, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 0.5)

    minmax = (0, 1, 0, 1)
    si = nap.compute_2d_mutual_info(dict_tc, features, minmax=minmax)
    np.testing.assert_approx_equal(si.loc[0, "SI"], 2.0)


@pytest.mark.parametrize(
    "tsd, expected_columns",
    [
        (nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 1))), [0]),
        (nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 2))), [0, 1]),
        (nap.Tsd(t=np.arange(0, 100), d=np.ones((100, ))), [0])
    ]
)
def test_compute_1d_tuning_curves_continuous(tsd, expected_columns):
    feature = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.arange(0, 100, 0.1) % 1.0)
    tc = nap.compute_1d_tuning_curves_continuous(tsd, feature, nb_bins=10)

    assert len(tc) == 10
    assert list(tc.columns) == expected_columns
    np.testing.assert_array_almost_equal(tc[0].values[1:], np.zeros(9))
    assert int(tc[0].values[0]) == 1.0


def test_compute_1d_tuning_curves_continuous_with_ep():
    tsdframe = nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 1)))
    feature = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.arange(0, 100, 0.1) % 1.0)
    ep = nap.IntervalSet(start=0, end=50)
    tc1 = nap.compute_1d_tuning_curves_continuous(tsdframe, feature, nb_bins=10)
    tc2 = nap.compute_1d_tuning_curves_continuous(tsdframe, feature, nb_bins=10, ep=ep)
    pd.testing.assert_frame_equal(tc1, tc2)


def test_compute_1d_tuning_curves_continuous_with_min_max():
    tsdframe = nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 1)))
    feature = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.arange(0, 100, 0.1) % 1.0)
    tc = nap.compute_1d_tuning_curves_continuous(
        tsdframe, feature, nb_bins=10, minmax=(0, 1)
    )
    assert len(tc) == 10
    np.testing.assert_array_almost_equal(tc[0].values[1:], np.zeros(9))
    assert tc[0].values[0] == 1.0

@pytest.mark.parametrize(
    "tsdframe, expected_columns",
    [
        (nap.TsdFrame(t=np.arange(0, 100), d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2))), [0, 1]),
        (
                nap.TsdFrame(t=np.arange(0, 100), d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
                             columns=["x", "y"]),
                ["x", "y"]
        ),
        (nap.Tsd(t=np.arange(0, 100), d=np.hstack((np.ones((100, )) * 2))), [0])

    ]
)
@pytest.mark.parametrize("nb_bins", [1, 2, 3])
def test_compute_2d_tuning_curves_continuous(nb_bins, tsdframe, expected_columns):

    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    tc, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, nb_bins)

    assert isinstance(tc, dict)
    assert list(tc.keys()) == expected_columns
    for i in tc.keys():
        assert tc[i].shape == (nb_bins, nb_bins)

    assert isinstance(xy, list)
    assert len(xy) == 2
    for i in range(2):
        assert np.min(xy) > 0
        assert np.max(xy) < 1


def test_compute_2d_tuning_curves_continuous_output_value():
    tsdframe = nap.TsdFrame(t=np.arange(0, 100), d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)))
    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    tc, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2)
    tmp = np.zeros((2, 2, 2))
    tmp[:, 0, 0] = [1, 2]
    for i in range(2):
        np.testing.assert_array_almost_equal(tc[i], tmp[i])


def test_compute_2d_tuning_curves_continuous_with_ep():
    tsdframe = nap.TsdFrame(
        t=np.arange(0, 100), d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2))
    )
    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    ep = nap.IntervalSet(start=0, end=7)
    tc1, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2, ep=ep)
    tc2, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2)

    for i in tc1.keys():
        np.testing.assert_array_almost_equal(tc1[i], tc2[i])






@pytest.mark.filterwarnings("ignore")
def test_compute_2d_tuning_curves_with_minmax():
    tsdframe = nap.TsdFrame(
        t=np.arange(0, 100), d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2))
    )
    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    minmax = (0, 10, 0, 2)
    tc, xy = nap.compute_2d_tuning_curves_continuous(
        tsdframe, features, 2, minmax=minmax
    )

    assert len(xy) == 2
    xbins = np.linspace(minmax[0], minmax[1], 3)
    np.testing.assert_array_almost_equal(xy[0], xbins[0:-1] + np.diff(xbins) / 2)
    ybins = np.linspace(minmax[2], minmax[3], 3)
    np.testing.assert_array_almost_equal(xy[1], ybins[0:-1] + np.diff(ybins) / 2)
