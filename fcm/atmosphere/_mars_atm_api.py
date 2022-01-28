"""Module for accessing the website http://www-mars.lmd.jussieu.fr/mcd_python/, which provides
height v. atmospheric density data for any pair of coordinates on Mars, for a given time stamp.
"""
__all__ = ["martian_atmosphere_api"]

import re
import io
import datetime
import requests
import jdcal
import pandas as pd

from fcm.models import _check_number

BASE_URL = "http://www-mars.lmd.jussieu.fr/mcd_python/"


###################################################
def martian_atmosphere_api(latitude, longitude, timestamp, zkey=2):
    """Loads atmospheric density data for any coordinates on Mars for a given timestamp.
    The timestamp is important, since the density data varies significantly with Martian seasons.
    
    Parameters
    ----------
    latitiude : float
        degrees North
        -90 <= latitude <= 90
    
    longitude : float
        degrees East
        -180 < longitude <= 180
        
    timestamp : Union[datetime.date, datetime.datetime]
        timestamp for which to request the data
        time zone = UTC

    zkey : int, optional
        Key for altitude definition (2 is default)
        1: xz is the radial distance from the center of the planet (km).
        2: xz is the altitude above the Martian zero datum (Mars geoid or “areoid”) (km).
        3: xz is the altitude above the local surface (km).
        4: xz is the pressure level (kPa).
        5: xz is the altitude above reference radius (3,396.106 km) (km).    

    Returns
    -------
    pandas.Series
        atmoshperic density (kg/m^3)
        index = (according to :param zkey)
    """

    # Define a dictionary of altitude types (2 is default)
    altitude_type = {1: "radial distance from the center of the planet (km)",
                     2: "altitude above MOLA_0 (km)",
                     3: "altitude above the local surface (km)",
                     4: "pressure level (kPa)",
                     5: "altitude above reference radius (km)"}

    # Check for valid inputs
    zkey = _check_number(zkey, "zkey", False, 1, True, 5, True)
    latitude = _check_number(latitude, "latitude", True, -90, True, 90, True)
    longitude = _check_number(longitude, "longitude", True, -180, False, 180, True)
    if not isinstance(timestamp, (datetime.date, datetime.datetime)):
        raise TypeError("timestamp must be a date or datetime object")
    
    url, params = _request_url(latitude, longitude, timestamp, zkey)
    txt_url = _get_txt_url(url, params)
    dataframe = _load_and_parse_txt_file(txt_url)

    # MCD provides results in m (or Pa); convert to km (kPa) and
    # set appropriate index name
    dataframe.index *= 1e-3 
    dataframe.index.name = altitude_type[zkey]
    
    return dataframe.iloc[:, 0]


###################################################
def _request_url(latitude, longitude, timestamp, zkey):
    """Convert latitude, longitude, timestamp and zkey into a request url to the website."""
    
    jdate = sum(jdcal.gcal2jd(timestamp.year, timestamp.month, timestamp.day))

    if isinstance(timestamp, datetime.datetime):
        jdate += timestamp.hour / 24 + timestamp.minute / (24*60) + timestamp.second / (24*3600) \
                 + timestamp.microsecond / (24*3600*1e6)

    # Construct URL using Julian date, lat, lon, altitude (and zkey)
    url = BASE_URL + "cgi-bin/mcdcgi.py"
    params = {'datekeyhtml': "0",
              'localtime': "0",
              'julian': f"{jdate:.5f}",
              'latitude': f"{latitude:.9f}",
              'longitude': f"{longitude:.9f}",
              'altitude': "all",
              'averaging': "off",  # turns off zonal averaging
              'hrkey': "1",  # uses high-res topography
              'zkey': f"{zkey:d}",
              'var1': "rho",  # var1 is density
              'colorm': "jet"}

    return url, params


###################################################
def _get_txt_url(url, params, timeout=4, pattern=re.compile("txt/[a-f0-9]+.txt")):
    """
    Sends GET request to url with params and timeout.
    
    Extracts url where data file can be downloaded from the response, returns it.
    """
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    print("html url:", response.url)
    
    match = pattern.search(response.text)
    if match is None:
        raise ValueError("pattern not found in html response:\n{}".format(response.text))

    return BASE_URL + match.group(0)


###################################################
def _load_and_parse_txt_file(url, timeout=2):
    """Loads csv file from url and converts it into a pandas.DataFrame"""
    
    print("data url:", url)
    buffer = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buffer.write(r.content)
    
    buffer.seek(0)
    atmosphere = pd.read_csv(buffer, sep=r"\s+", header=None, index_col=0, comment="#")
    buffer.close()
    
    assert atmosphere.shape[1] == 1,\
        "expected only two columns, got {:d}".format(atmosphere.shape[1] + 1)
    atmosphere.columns = ["Density (kg/m3)"]
    atmosphere.dropna(axis=0, how="any", inplace=True)
    
    return atmosphere
