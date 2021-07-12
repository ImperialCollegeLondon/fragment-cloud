import os, sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.path[0] != top_dir:
    sys.path.insert(0, top_dir)

import numpy as np
import pandas as pd

import fcm
import fcm.atmosphere as atm


###################################################
def setup(fixed_step=False, precision=1e-2, solver="AB2", cloud_model="CRM", pancake=True):
    
    # 1. Atmosphere
    rho_a = atm.US_standard_atmosphere()
    
    # 2. Model
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model=cloud_model,
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
def test_pancake_meteoroid_analysis(generate_plots=False):
    
    h_start = 100
    timesteps = [4e-3, 1e-3, 4e-4, 1e-4, 4e-5]
    precisions = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 2e-7]
    
    for cloud_model in fcm.CloudDispersionModel:
        # Reference Solution (fixed, very small step, RK4)
        sim1, meteoroid1 = setup(True, 1e-5, "RK4", cloud_model)
        results = fcm.simulate_impact(sim1, meteoroid1, h_start, craters=False, dedz=True)
        solution = results.energy_deposition

        errors = dict()
        for ode_solver in fcm.Timestepper:
            errors[ode_solver] = {"fixed": dict(), "adaptive": dict()}
            for dt in timesteps:
                sim, meteoroid = setup(True, dt, ode_solver, cloud_model)
                results = fcm.simulate_impact(sim, meteoroid, h_start, craters=False, dedz=True,
                                                timeseries=True)
                result = results.energy_deposition
                
                assert(len(results.clouds) == 1)
                num_steps = results.bulk.timeseries.index.size \
                    + next(iter(results.clouds.values())).timeseries.index.size
                
                pd.testing.assert_index_equal(result.index, solution.index)
                errors[ode_solver]["fixed"][num_steps] = np.linalg.norm(result - solution)
            
            for prec in precisions:
                sim, meteoroid = setup(False, prec, ode_solver, cloud_model)
                results = fcm.simulate_impact(sim, meteoroid, h_start, craters=False, dedz=True,
                                              timeseries=True)
                result = results.energy_deposition
                
                assert(len(results.clouds) == 1)
                num_steps = results.bulk.timeseries.index.size \
                    + next(iter(results.clouds.values())).timeseries.index.size
                
                pd.testing.assert_index_equal(result.index, solution.index)
                errors[ode_solver]["adaptive"][num_steps] = np.linalg.norm(result - solution)

        for ode_solver, results in errors.items():
            errors_fixed = pd.Series(results["fixed"])
            errors_adaptive = pd.Series(results["adaptive"])
            polyfit_fixed = np.polyfit(np.log(errors_fixed.index),
                                       np.log(errors_fixed.to_numpy()), 1)
            polyfit_adaptive = np.polyfit(np.log(errors_adaptive.index),
                                          np.log(errors_adaptive.to_numpy()), 1)
            
            np.testing.assert_allclose(polyfit_adaptive[0], polyfit_fixed[0], 2e-1)
            assert polyfit_fixed[0] < -0.9
            
            if ode_solver is not fcm.Timestepper.forward_euler:
                advantage_adaptive = np.exp(np.polyval(polyfit_fixed, np.log(1e4)) \
                    - np.polyval(polyfit_adaptive, np.log(1e4)))
                assert advantage_adaptive > 2,\
                    "adaptive advantage = {:.2f}".format(advantage_adaptive)
        
        if generate_plots:
            import matplotlib.pyplot as plt     
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 7))
            axes = axes.flatten()

            for i, (ode_solver, results) in enumerate(errors.items()):
                for label, data in results.items():
                    axes[i].loglog(list(data.keys()), list(data.values()), label=label)
                axes[i].set_title(ode_solver)
                axes[i].set_ylabel("error")
                axes[i].set_xlabel("iterations")
                axes[i].legend(loc='best')

            fig.suptitle("Convergence analysis for {} cloud model".format(cloud_model.value))
            plt.show()
    
    
###################################################
if __name__ == "__main__":
    test_pancake_meteoroid_analysis(True)
