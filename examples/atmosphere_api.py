import os, sys
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path[0] != BASE_PATH:
    sys.path.insert(0, BASE_PATH)

from datetime import datetime
import matplotlib.pyplot as plt
import fcm.atmosphere as atm

if __name__ == "__main__":
    lat_olympus_mons, long_olympus_mons = 18.65, 226.2 - 360
    lat_hellas_planitia, long_hellas_planitia = -42.4, 70.5
    
    date = datetime.now()
    
    rho_a_olympus = atm.martian_atmosphere_api(lat_olympus_mons, long_olympus_mons, date)
    rho_a_hellas = atm.martian_atmosphere_api(lat_hellas_planitia, long_hellas_planitia, date)
    
    plt.plot(rho_a_olympus.to_numpy(), rho_a_olympus.index.to_numpy(),
             label="Olympus Mons (18.65 deg N, 226.2 deg E)")
    plt.plot(rho_a_hellas.to_numpy(), rho_a_hellas.index.to_numpy(),
             label="Hellas Planitia (42.4 deg S, 70.5 deg E)")
    plt.plot(rho_a_olympus.iloc[0], rho_a_olympus.index[0], 'bo')
    plt.plot(rho_a_hellas.iloc[0], rho_a_hellas.index[0], 'go')
    plt.xscale("log")
    plt.xlabel("atmospheric density (kg/m^3)")
    plt.ylabel("altitiude above MOLA_0")
    plt.legend()
    plt.show()
