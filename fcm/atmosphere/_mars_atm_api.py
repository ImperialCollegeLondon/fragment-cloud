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
def martian_atmosphere_api(latitude, longitude, timestamp):
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
    
    Returns
    -------
    pandas.Series
        atmoshperic density (kg/m^3)
        index = altitude above MOLA_0 (km)
    """
    latitude = _check_number(latitude, "latitude", True, -90, True, 90, True)
    longitude = _check_number(longitude, "longitude", True, -180, False, 180, True)
    if not isinstance(timestamp, (datetime.date, datetime.datetime)):
        raise TypeError("timestamp must be a date or datetime object")
    
    url = _request_url(latitude, longitude, timestamp)
    txt_url = _get_txt_url(url)
    dataframe = _load_and_parse_txt_file(txt_url)
    
    dataframe.index *= 1e-3
    dataframe.index.name = "altitude above MOLA_0 (km)"
    
    return dataframe.iloc[:, 0]


###################################################
def _request_url(latitude, longitude, timestamp):
    """Converts latitude, longitude and timestamp into a request url to the website"""
    
    jdate = sum(jdcal.gcal2jd(timestamp.year, timestamp.month, timestamp.day))

    if isinstance(timestamp, datetime.datetime):
        jdate += timestamp.hour / 24 + timestamp.minute / (24*60) + timestamp.second / (24*3600)
    
    url = BASE_URL + "cgi-bin/mcdcgi.py?"
    url += "&julian={:.5f}&latitude={:.9f}&longitude={:.9f}".format(jdate, latitude, longitude)
    url += "&altitude=all&zkey=2&var1=rho&colorm=jet"
    
    return url


###################################################
def _get_txt_url(url, timeout=4, pattern=re.compile("txt/[a-f0-9]+.txt")):
    """Sends GET request to url with timeout. Extracts url where data file can be downloaded from
    the response, returns it.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    match = pattern.search(response.text)
    if match is None:
        raise ValueError("pattern not found in html response:\n{}".format(response.text))

    return BASE_URL + match.group(0)


###################################################
def _load_and_parse_txt_file(url, timeout=2):
    """Loads csv file from url and converts it into a pandas.DataFrame"""
    
    print("txt url:", url)
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
    atmosphere.index.name = "altitude above MOLA_0 (m)"
    
    atmosphere.dropna(axis=0, how="any", inplace=True)
    
    return atmosphere
