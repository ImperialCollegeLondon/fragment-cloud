__all__ = ["martian_atmosphere_api"]

import re
import io
import datetime
import requests
import jdcal
import pandas as pd

def _request_url(latitude, longitude, timestamp):
    
    if not isinstance(timestamp, (datetime.date, datetime.datetime)):
        raise TypeError("timestamp must be a date or datetime object")
    
    jdate = sum(jdcal.gcal2jd(timestamp.year, timestamp.month, timestamp.day))

    if isinstance(timestamp, datetime.datetime):
        jdate += timestamp.hour / 24 + timestamp.minute / (24*60) + timestamp.second / (24*3600)
    
    url = "http://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py?"
    url += "&julian={:.5f}&latitude={:.9f}&longitude={:.9f}".format(jdate, latitude, longitude)
    url += "&altitude=all&zkey=2&var1=rho&colorm=jet"
    
    return url


def _get_html_response(url, timeout=2):
    
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError("request failed with status code {:d}".format(response.status_code))
    
    return response.text


def _extract_txt_url(html, pattern=re.compile("txt/[a-f0-9]+.txt")):
    
    match = pattern.search(html)
    if match is None:
        raise ValueError("pattern not found")

    return "http://www-mars.lmd.jussieu.fr/mcd_python/{}".format(match.group(0))


def _load_and_parse_txt_file(url, timeout=2):
    
    buffer = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buffer.write(r.content)
    
    buffer.seek(0)
    atmosphere = pd.read_csv(buffer, sep="\s+", header=None, index_col=0, comment="#")
    buffer.close()
    
    assert atmosphere.shape[1] == 1,\
        "expected only two columns, got {:d}".format(atmosphere.shape[1] + 1)
    atmosphere.columns = ["Density (kg/m3)"]
    atmosphere.index.name = "altitude above MOLA_0 (m)"
    
    atmosphere.dropna(axis=0, how="any", inplace=True)
    
    return atmosphere
    

def martian_atmosphere_api(latitude, longitude, timestamp):
    
    url = _request_url(latitude, longitude, timestamp)
    response = _get_html_response(url)
    txt_url = _extract_txt_url(response)
    dataframe = _load_and_parse_txt_file(txt_url)
    
    dataframe.index *= 1e-3
    dataframe.index.name = "altitude above MOLA_0 (km)"
    
    return dataframe.iloc[:, 0]
