"""Unit tests for fcm.atmosphere module"""

import os, sys
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.path[0] != BASE_PATH:
    sys.path.insert(0, BASE_PATH)

import datetime
import numpy as np
import pandas as pd

import fcm.atmosphere as atm

def test_us_atmosphere():
    atmosphere = atm.US_standard_atmosphere()
    assert isinstance(atmosphere, pd.Series)
    assert atmosphere.size > 10
    assert not atmosphere.isnull().any()
    assert (np.diff(atmosphere.index) > 0).all()
    assert (np.diff(atmosphere.to_numpy()) < 0).all()
    

def test_mars_atmosphere():
    atmosphere = atm.static_martian_atmosphere()
    assert isinstance(atmosphere, pd.Series)
    assert atmosphere.size > 10
    assert not atmosphere.isnull().any()
    assert (np.diff(atmosphere.index) > 0).all()
    assert (np.diff(atmosphere.to_numpy()) < 0).all()


def test_exp_atmosphere():
    rho0 = 1
    H = 2
    size = 50
    atmosphere = atm.exponential(rho0, 10, H, size)
    
    assert isinstance(atmosphere, pd.Series)
    assert atmosphere.size == 50
    assert not atmosphere.isnull().any()
    assert (np.diff(atmosphere.index) > 0).all()
    assert (np.diff(atmosphere.to_numpy()) < 0).all()
    np.testing.assert_allclose((atmosphere * np.exp(atmosphere.index / 2)).to_numpy(), 1)


def test_mars_api():
    timestamp = datetime.datetime(2020, 5, 16, 6, 33, 24)
    lat = -72.176
    long = -2.956
    
    atmosphere = atm.martian_atmosphere_api(lat, long, timestamp)
    assert isinstance(atmosphere, pd.Series)
    assert atmosphere.size > 10
    assert not atmosphere.isnull().any()
    assert (np.diff(atmosphere.index) > 0).all()
    assert (np.diff(atmosphere.to_numpy()) < 0).all()


if __name__ == "__main__":
    test_us_atmosphere()
    test_mars_atmosphere()
    test_exp_atmosphere()
    test_mars_api()
