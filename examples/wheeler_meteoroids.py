import os, sys
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

import fcm
import fcm.atmosphere as atm

from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###################################################
def chelyabinsk_meteoroid(atmosphere, precision=1e-2):
    groups = [
        fcm.StructuralGroup(mass_fraction=0.25e-2, density=3.3e3, strength=600, pieces=1,
                            cloud_mass_frac=0.75, strength_scaler=0.08, fragment_mass_fractions=(0.6, 0.4)),
        fcm.StructuralGroup(mass_fraction=0.93, density=3.3e3, strength=1480, pieces=1,
                            cloud_mass_frac=0.75, strength_scaler=0.3, fragment_mass_fractions=(0.6, 0.4)),
        fcm.StructuralGroup(mass_fraction=0.8e-2, density=3.3e3, strength=1750, pieces=8,
                            cloud_mass_frac=0.75, strength_scaler=0.07, fragment_mass_fractions=(0.6, 0.4)),
        fcm.StructuralGroup(mass_fraction=2.55e-2, density=3.3e3, strength=2500, pieces=10,
                            cloud_mass_frac=1, strength_scaler=None, fragment_mass_fractions=(0.6, 0.4)),
        fcm.StructuralGroup(mass_fraction=1e-2, density=3.3e3, strength=3500, pieces=6,
                            cloud_mass_frac=1, strength_scaler=None, fragment_mass_fractions=(0.6, 0.4)),
        fcm.StructuralGroup(mass_fraction=2.4e-2, density=3.3e3, strength=1.56e4, pieces=2,
                            cloud_mass_frac=0.75, strength_scaler=0.07, fragment_mass_fractions=(0.6, 0.4))
    ]
    meteoroid = fcm.FCMmeteoroid(19.16, 18.3, 2.5e3, 19.8/2, 600, 0, groups)
    
    simulation = fcm.FragmentCloudModel(9.81, 6371, atmosphere, ablation_coeff=7e-9,
                                        cloud_disp_coeff=1.8/3.5, strengh_scaling_disp=0,
                                        fragment_mass_disp=0, precision=precision)
    return meteoroid, simulation


###################################################
def kosice_meteoroid(atmosphere, precision=1e-2):
    groups = [fcm.StructuralGroup(0.14, 3.4e3, 2, 300, 2e-3, 2, (0.8, 0.2)),
              fcm.StructuralGroup(0.03, 3.4e3, 35, 1, 0.5, 0.1, (0.8, 0.2)),
              fcm.StructuralGroup(0.03, 3.4e3, 55, 1, 0.5, 0.8, (0.8, 0.2)),
              fcm.StructuralGroup(0.8, 3.4e3, 1150, 1, 0.5, 0.3, (0.8, 0.2))]
    
    meteoroid = fcm.FCMmeteoroid(15, 60, 2.5e3, 1.388/2, 2, 0, groups)
    
    simulation = fcm.FragmentCloudModel(9.81, 6371, atmosphere, ablation_coeff=1e-8,
                                        cloud_disp_coeff=2/3.5, strengh_scaling_disp=0,
                                        fragment_mass_disp=0, precision=precision)
    return meteoroid, simulation


###################################################
def benesov_meteoroid(atmosphere, precision=1e-2):
    groups = [fcm.StructuralGroup(1e-3, 3.2e3, 25, 1, 0.5, 0.4, (0.8, 0.2)),
              fcm.StructuralGroup(1e-3, 3.2e3, 55, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(1e-3, 3.2e3, 80, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(1e-3, 3.2e3, 100, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(1.5e-3, 3.2e3, 150, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(0.5e-3, 3.2e3, 250, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(2e-3, 3.2e3, 700, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(2e-3, 3.2e3, 900, 1, 0.5, 0.2, (0.8, 0.2)),
              fcm.StructuralGroup(0.01, 3.2e3, 1.1e3, 1, 0.5, 0.2, (0.8, 0.2)),
              fcm.StructuralGroup(0.02, 3.2e3, 1.5e3, 1, 0.5, 0.2, (0.8, 0.2)),
              fcm.StructuralGroup(0.07, 3.2e3, 2.1e3, 1, 0.7, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(0.34, 3.2e3, 2.9e3, 1, 0.5, 0.3, (0.8, 0.2)),
              fcm.StructuralGroup(0.05, 3.2e3, 3.8e3, 1, 0.5, 0.2, (0.8, 0.2)),
              fcm.StructuralGroup(0.15, 3.2e3, 6.1e3, 1, 0.5, 0.2, (0.8, 0.2)),
              fcm.StructuralGroup(0.35, 3.2e3, 17e3, 1, 1, None, None)]
    
    meteoroid = fcm.FCMmeteoroid(21.5, 81, 3.2e3, 1.348/2, 20, 0, groups)
    
    simulation = fcm.FragmentCloudModel(9.81, 6371, atmosphere, ablation_coeff=1e-8,
                                        cloud_disp_coeff=2/3.5, strengh_scaling_disp=0,
                                        fragment_mass_disp=0, precision=precision)
    return meteoroid, simulation


###################################################
def tagish_lake_meteoroid(atmosphere, precision=1e-2):
    groups = [fcm.StructuralGroup(0.01, 1.64e3, 1, 10, 0.2, 0.1, (0.5, 0.5)),
              fcm.StructuralGroup(5e-3, 1.64e3, 1.8, 5, 0.2, 0.1, (0.5, 0.5)),
              fcm.StructuralGroup(8e-3, 1.64e3, 3.8, 5, 0.2, 0.15, (0.5, 0.5)),
              fcm.StructuralGroup(0.012, 1.64e3, 6, 2, 0.2, 0.2, (0.5, 0.5)),
              fcm.StructuralGroup(0.01, 1.64e3, 15, 1, 0.2, 0.2, (0.5, 0.5)),
              fcm.StructuralGroup(7.5e-3, 1.64e3, 40, 1, 0.6, 0.4, (0.5, 0.5)),
              fcm.StructuralGroup(0.01, 1.64e3, 90, 1, 0.6, 0.1, (0.5, 0.5)),
              fcm.StructuralGroup(0.046, 1.64e3, 140, 1, 0.8, 0.3, (0.5, 0.5)),
              fcm.StructuralGroup(0.028, 1.64e3, 350, 1, 0.2, 0.25, (0.5, 0.5)),
              fcm.StructuralGroup(0.1, 1.64e3, 900, 1, 0.99, 0.05, (0.5, 0.5)),
              fcm.StructuralGroup(0.21, 1.64e3, 1.21e3, 1, 0.99, 0.05, (0.5, 0.5)),
              fcm.StructuralGroup(0.0725, 1.64e3, 1.82e3, 4, 0.99, 0.05, (0.5, 0.5)),
              fcm.StructuralGroup(0.369, 1.64e3, 2.5e3, 1, 1, None, None),
              fcm.StructuralGroup(0.1, 1.64e3, 3.3e3, 1, 1, None, None),
              fcm.StructuralGroup(0.012, 1.64e3, 3.6e3, 1, 1, None, None)]
    
    meteoroid = fcm.FCMmeteoroid(15.8, 17.8, 1.64e3, 4.5/2, 0.5, 0, groups)
    
    simulation = fcm.FragmentCloudModel(9.81, 6371, atmosphere, ablation_coeff=1e-8,
                                        cloud_disp_coeff=1/3.5, strengh_scaling_disp=0,
                                        fragment_mass_disp=0, precision=precision)
    return meteoroid, simulation


###################################################
class Event(Enum):
    chelyabinsk = chelyabinsk_meteoroid
    kosice = kosice_meteoroid
    benesov = benesov_meteoroid
    tagish_lake = tagish_lake_meteoroid
    

###################################################
file_names = {Event.chelyabinsk: "ChelyabinskEnergyDep_Wheeler-et-al-2018.txt",
              Event.kosice: "KosiceEnergyDep_Wheeler-et-al-2018.txt",
              Event.benesov: "BenesovEnergyDep_Wheeler-et-al-2018.txt",
              Event.tagish_lake: "TagishLakeEnergyDep_Wheeler-et-al-2018.txt"}


###################################################
def read_data(event):
    
    data = pd.read_csv(os.path.join(THIS_DIR, "data", file_names[event]),
                       sep='\t', header=0, index_col=0)
    
    data.columns = ["dEdz [kt TNT / km]", "min. dEdz [kt TNT / km]", "max. dEdz [kt TNT / km]"]
    data.index.name = "altitude [km]"
    
    return data
    

###################################################
if __name__ == "__main__":
    
    evt = Event.kosice
    obs = read_data(evt)
    
    for i, (label, series) in enumerate(obs.items()):
        plt.plot(series.to_numpy(), series.index.to_numpy(), '-' if i == 0 else '--', label=label)
    
    plt.xlabel("dEdz [kt TNT / km]")
    plt.ylabel("altitude [km]")
    plt.xscale('log')
    plt.legend(loc='best')
    plt.show()
    
    rho_a = atm.US_standard_atmosphere()
    impactor, sim = evt(rho_a, 1e-4)
    sim.simulate_impact(impactor, 100, craters=False, dedz=True, final_states=True)
    
    mask = np.logical_and(sim.results.energy_deposition.index.to_numpy() >= obs.index.min(),
                          sim.results.energy_deposition.index.to_numpy() <= obs.index.max())
    
    plt.plot(sim.results.energy_deposition.to_numpy()[mask],
             sim.results.energy_deposition.index.to_numpy()[mask], label='fcm')
    plt.plot(obs['min. dEdz [kt TNT / km]'].to_numpy(), obs.index.to_numpy(), "--", label='observation (min)')
    plt.plot(obs['max. dEdz [kt TNT / km]'].to_numpy(), obs.index.to_numpy(), "--", label='observation (max)')
    
    plt.xlabel("dEdz [kt TNT / km]")
    plt.ylabel("altitude [km]")
    plt.xscale('log')
    plt.legend(loc='best')
    plt.show()
    