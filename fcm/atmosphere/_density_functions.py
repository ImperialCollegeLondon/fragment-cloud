"""Provides a couple of standard atmoshperic density tables"""
__all__ = ["exponential", "US_standard_atmosphere", "static_martian_atmosphere"]

import os
import math
import numpy as np
import pandas as pd

from fcm.models import _check_number

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


###################################################
def exponential(rho0, hmax, scale_height, granularity=None, h0=0):
    """Computes exponentially decaying atmoshperic density
    rho(h) = rho0 * exp(-h / H)
    
    Parameters
    ----------
    rho0 : float
        atmospheric density at altitude h0 (default: h0 = 0)
        rho0 > 0
    
    hmax : float
        altitude in [km] up to which to calculate the atmospheric density
    
    scale_height : float
        height difference H for which rho(h + H) = rho(h) / e
    
    granularity : int, optional
        number of points at which to calculate the atmospheric density
        granularity >= 2
        default : granularity = (hmax - hmin) / scale_height
    
    h0 : float, optional
        altitude in [km] where rho(h0) = rho0
        default = 0
        h0 < hmax
    
    Returns
    -------
    pandas.Series
        atmospheric density (kg/m^3)
        index = altitude above 0 (km)
    """
    rho0 = _check_number(rho0, "rho0", lower_bound=0)
    h0 = _check_number(h0, "h0")
    hmax = _check_number(hmax, "hmax")
    if not h0 < hmax:
        raise ValueError("hmax ({:.2f}) must be larger than h0 ({:.2f})".format(hmax, h0))
    
    if granularity is None:
        granularity = math.ceil((hmax - h0) / scale_height) + 1
    else:
        granularity = _check_number(granularity, "granularity", False, 2, True)
    
    h = np.linspace(h0, hmax, granularity)
    
    return pd.Series(data=rho0 * np.exp(-h / scale_height), index=h)


###################################################
def US_standard_atmosphere():
    """Returns the US Standard Atmosphere table (1976),
    National Aeronautics and Space Administration
    
    Returns
    -------
    pandas.Series
        atmospheric density (kg/m^3)
        index = altitude above sea level (km)
    """
    data = pd.read_csv(os.path.join(THIS_DIR, "AltitudeDensityTable.csv"),
                       sep=" ", header=None, skiprows=6, index_col=0)
    
    data.columns = ["Atmospheric Density (kg/m3)", "H - scale height (m)"]
    data.index.name = "Altitude (km)"
    data.index /= 1e3
    
    return data["Atmospheric Density (kg/m3)"]


###################################################
def static_martian_atmosphere():
    """Returns example of a Martian atmospheric density. Intended for testing and debugging.
    Use the fcm.atmosphere.martian_atmosphere_api() function for accurate data.
    
    Returns
    -------
    pandas.Series
        atmospheric density (kg/m^3)
        index = altitiude above MOLA_0 (km)
    """
    data = pd.read_csv(os.path.join(THIS_DIR, "Martian_Density.csv"),
                       sep="\t", header=None, index_col=0)
    
    data = data[data.columns[0]]
    data.name = "Atmospheric Density (kg/m3)"
    data.index.name = "Altitude (km)"
    
    return data
