import os, sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fcm


###################################################
def dummy_atmosphere(rho0=1):
    h = np.linspace(0, 100, 11)
    
    return pd.Series(data=rho0*np.exp(-h/8), index=h)


###################################################
def test_pancake_meteoroid():
    
    rho_a = dummy_atmosphere()
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
    
    breakpoint()
    

###################################################
def test_fragmenting_meteoroid():
    
    rho_a = dummy_atmosphere(0.01)
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="DCM",
                              timestepper="AB2", precision=1e-2, strengh_scaling_disp=0.4,
                              fragment_mass_disp=0, dh=1)
    groups = [fcm.StructuralGroup(mass_fraction=1, pieces=3, density=3e3, strength=500)]
    impactor = fcm.FCMmeteoroid(velocity=10, angle=30, density=3e3, strength=100, radius=1,
                                structural_groups=groups)
    
    results = fcm.simulate_impact(model, impactor, h_start=100, craters=True, final_states=True,
                                  dedz=True, seed=12)
    if results.craters is not None:
        x_min = (results.craters.x - results.craters.r).min()
        x_max = (results.craters.x + results.craters.r).max()
        x_margin = 0.05 * (x_max - x_min)
        
        y_min = (results.craters.y - results.craters.r).min()
        y_max = (results.craters.y + results.craters.r).max()
        y_margin = 0.05 * (y_max - y_min)
        
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(x_min - x_margin, x_max + x_margin)
        ax.set_xlim(y_min - y_margin, y_max + y_margin)
        
        for row in results.craters.itertuples():
            circle = plt.Circle((row.y, row.x), row.r, color='black', fill=False)
            ax.add_artist(circle)
        
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_aspect('equal')
        plt.show()
    else:
        print("No craters were formed")
    
    if results.energy_deposition is not None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(results.energy_deposition.to_numpy(), results.energy_deposition.index.to_numpy())
        ax.set_ylabel("height [km]")
        ax.set_xlabel("dE/dz [kt TNT / km]")
        # ax.set_xscale('log')
        plt.show()
    
    breakpoint()


####################################################################################################
if __name__ == "__main__":
    test_pancake_meteoroid()
    test_fragmenting_meteoroid()
