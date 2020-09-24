import os, sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fcm
import fcm.atmosphere as atm


###################################################
def setup(fixed_step=False, precision=1e-2, solver="AB2", pancake=True):
    
    # 1. Atmosphere
    rho_a = atm.US_standard_atmosphere()
    
    # 2. Model
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="CRM",
                              timestepper=solver, precision=precision, variable_timestep=not fixed_step)
    # 3. Impactor
    if pancake:
        impactor = fcm.PancakeMeteoroid(velocity=20, angle=40, density=3.3e3, radius=10, 
                                        strength=100)
    else:
        impactor = fcm.FragmentationMeteoroid(velocity=20, angle=40, density=3.3e3, radius=0.2,
                                              strength=1000)
    return model, impactor


###################################################
def pancake_meteoroid_analysis():
    
    h_start = 100
    timesteps = {True: [4e-3, 1e-3, 4e-4, 1e-4, 4e-5], False: [4e-3, 1e-3, 4e-4]}
    precisions = {True: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 2e-7], False: [1e-2, 1e-3, 1e-4, 1e-5]}
    
    for pancake_meteoroid in [True, False]:
        # Reference Solution (fixed, very small step, RK4)
        sim1, meteoroid1 = setup(True, 6e-6 if pancake_meteoroid else 1e-3, "RK4", pancake_meteoroid)
        results = fcm.simulate_impact(sim1, meteoroid1, h_start, craters=False, dedz=True)
        solution = results.energy_deposition
        
        errors = dict()
        for ode_solver in fcm.Timestepper:
            errors[ode_solver.value] = {"fixed": dict(), "adaptive": dict()}
            for dt in timesteps[pancake_meteoroid]:
                sim, meteoroid = setup(True, dt, ode_solver, pancake_meteoroid)
                results = fcm.simulate_impact(sim, meteoroid, h_start, craters=False, dedz=True,
                                              timeseries=pancake_meteoroid)
                result = results.energy_deposition
                
                if pancake_meteoroid:
                    assert(len(results.clouds) == 1)
                    num_steps = results.bulk.timeseries.index.size \
                        + next(iter(results.clouds.values())).timeseries.index.size
                else:
                    num_steps = dt
                
                pd.testing.assert_index_equal(result.index, solution.index)
                errors[ode_solver.value]["fixed"][num_steps] = np.linalg.norm(result - solution)
            
            for prec in precisions[pancake_meteoroid]:
                sim, meteoroid = setup(False, prec, ode_solver, pancake_meteoroid)
                results = fcm.simulate_impact(sim, meteoroid, h_start, craters=False, dedz=True,^
                                              timeseries=pancake_meteoroid)
                result = results.energy_deposition
                
                if pancake_meteoroid:
                    assert(len(results.clouds) == 1)
                    num_steps = results.bulk.timeseries.index.size \
                        + next(iter(results.clouds.values())).timeseries.index.size
                else:
                    num_steps = prec
                
                pd.testing.assert_index_equal(result.index, solution.index)
                (result - solution).plot()
                plt.show()
                errors[ode_solver.value]["adaptive"][num_steps] = np.linalg.norm(result - solution)
        
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 7))
        axes = axes.flatten()
        
        for i, (ode_solver, results) in enumerate(errors.items()):
            for label, data in results.items():
                axes[i].loglog(list(data.keys()), list(data.values()), label=label)
            axes[i].set_title(ode_solver)
            axes[i].set_ylabel("error")
            axes[i].set_xlabel("iterations")
            axes[i].legend(loc='best')
        
        fig.suptitle("Convergence analysis for {} meteoroid".format(
            "pancake" if pancake_meteoroid else "fragmenting"
        ))
        plt.show()
    
    
###################################################
if __name__ == "__main__":
    
    pancake_meteoroid_analysis()
