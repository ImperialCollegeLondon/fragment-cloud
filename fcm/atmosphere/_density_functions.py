__all__ = ["exponential", "US_standard_atmosphere", "static_martian_atmosphere"]

import os
import math
import numpy as np
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def exponential(rho0, hmax=100, scale_height=8, hmin=0):
    h = np.linspace(hmin, hmax, math.ceil((hmax - hmin) / scale_height) + 1)
    
    return pd.Series(data=rho0 * np.exp(-h / scale_height), index=h)


def US_standard_atmosphere():
    data = pd.read_csv(os.path.join(THIS_DIR, "AltitudeDensityTable.csv"), sep=" ", header=None, skiprows=6, index_col=0)
    
    data.columns = ["Atmospheric Density (kg/m3)", "H - scale height (m)"]
    data.index.name = "Altitude (km)"
    data.index /= 1e3
    
    return data["Atmospheric Density (kg/m3)"]


def static_martian_atmosphere():
    data = pd.read_csv(os.path.join(THIS_DIR, "Martian_Density.csv"), sep="\t", header=None, index_col=0)
    
    data = data[data.columns[0]]
    data.name = "Atmospheric Density (kg/m3)"
    data.index.name = "Altitude (km)"
    
    return data
