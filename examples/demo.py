import os, sys
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path[0] != BASE_PATH:
    sys.path.insert(0, BASE_PATH)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fcm
import fcm.atmosphere as atm
from fcm import crater_tools


###################################################
def pancake_meteoroid():
    
    rho_a = atm.exponential(1, 100, 8, 11)
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="CRM",
                              timestepper="AB2", precision=1e-1)
    impactor = fcm.PancakeMeteoroid(velocity=20, angle=40, density=3.3e3, radius=10, strength=100)
    
    results = fcm.simulate_impact(model, impactor, h_start=100, craters=False, timeseries=True)
    bulk = results.bulk
    
    assert len(results.fragments) == 0
    
    if len(results.clouds) > 0:
        assert len(results.clouds) == 1
        cloud = next(iter(results.clouds.values()))
        combined = pd.concat([bulk.timeseries, cloud.timeseries])
    else:
        combined = bulk.timeseries
    
    plt.plot(combined.dEdz, combined.z, '+')
    if len(results.clouds) > 0:
        plt.plot(combined.r/combined.r.max() * combined.dEdz.max(), combined.z)
    plt.ylabel("height [km]")
    plt.xlabel("dE/dz [kt TNT / km]")
    plt.show()
    
    t_diff = np.diff(combined.index).astype(float)*1e-9
    r_diff = np.diff(combined.r) / t_diff
    x_axis = combined.z.to_numpy()[1:]
    plt.plot(x_axis, t_diff, '+')
    plt.plot(x_axis, r_diff, 'x')
    plt.yscale('log')
    plt.show()
    

###################################################
def fragmenting_meteoroid():
    
    rho_a = atm.exponential(0.01, 100, 8, 11)
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="DCM",
                              timestepper="AB2", precision=1e-2, strengh_scaling_disp=0.4,
                              fragment_mass_disp=0, dh=1)
    groups = [fcm.StructuralGroup(mass_fraction=1, pieces=3, density=3e3, strength=500)]
    impactor = fcm.FCMmeteoroid(velocity=10, angle=30, density=3e3, strength=100, radius=1,
                                structural_groups=groups)
    
    results = fcm.simulate_impact(model, impactor, h_start=100, craters=True, final_states=True,
                                  dedz=True, seed=12)
    if results.craters is not None:
        crater_tools.plot_craters(results.craters)
    else:
        print("No craters were formed")
    
    if results.energy_deposition is not None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(results.energy_deposition.to_numpy(), results.energy_deposition.index.to_numpy())
        ax.set_ylabel("height [km]")
        ax.set_xlabel("dE/dz [kt TNT / km]")
        ax.set_xscale('log')
        plt.show()


####################################################################################################
if __name__ == "__main__":
    pancake_meteoroid()
    fragmenting_meteoroid()
