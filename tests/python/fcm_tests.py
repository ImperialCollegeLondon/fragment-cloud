import os, sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.path[0] != top_dir:
    sys.path.insert(0, top_dir)

import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import fcm
import fcm.atmosphere as atm


def test_dedz():
    rho_a = atm.exponential(1, 100, 8, 11)
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="CRM",
                              timestepper="AB2", precision=1e-3)
    impactor = fcm.PancakeMeteoroid(velocity=20, angle=40, density=3.3e3, radius=10, strength=100)
    
    results = fcm.simulate_impact(model, impactor, h_start=100, craters=False, timeseries=True)
    bulk = results.bulk
    
    assert len(results.fragments) == 0
    assert len(results.clouds) == 1
    cloud = next(iter(results.clouds.values()))
    combined = pd.concat([bulk.timeseries, cloud.timeseries])
    
    interpolation_function = interp1d(results.energy_deposition.index.to_numpy(),
                                      results.energy_deposition.to_numpy())
    interpolated_values = interpolation_function(combined.z.to_numpy())
    real_values = combined.dEdz.to_numpy()
    
    np.testing.assert_allclose(interpolated_values[1:-4], real_values[1:-4], rtol=5e-2)


def test_ablation_model():
    rho_a = atm.exponential(1, 100, 8, 11)
    model_std = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="CRM",
                                  timestepper="AB2", precision=1e-3)
    model_const_r = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="CRM",
                                      timestepper="AB2", precision=1e-3, ablation_model='constant_r')
    
    impactor = fcm.PancakeMeteoroid(velocity=20, angle=40, density=3.3e3, radius=10, strength=100)
    results_std = fcm.simulate_impact(model_std, impactor, h_start=100, craters=False, timeseries=True)
    results_const_r = fcm.simulate_impact(model_const_r, impactor, h_start=100, craters=False, timeseries=True)

    # make sure there is some difference, i.e. we're not running the exact same simulation twice
    assert (results_std.energy_deposition - results_const_r.energy_deposition).abs().max() > 1e-8
    
    # it's a big meteoroid, so assert results are very similar
    pd.testing.assert_series_equal(results_std.energy_deposition, results_const_r.energy_deposition, rtol=1e-3)


def test_fragmentation():
    
    rho_a = atm.exponential(0.01, 100, 8, 11)
    model = fcm.FCMparameters(g0=9.81, Rp=6371, atmospheric_density=rho_a, cloud_disp_model="DCM",
                              timestepper="AB2", precision=1e-2, strengh_scaling_disp=0.4,
                              fragment_mass_disp=0, dh=1)
    groups = [fcm.StructuralGroup(mass_fraction=1, pieces=3, density=3e3, strength=200)]
    impactor = fcm.FCMmeteoroid(velocity=10, angle=50, density=3e3, strength=100, radius=0.1,
                                structural_groups=groups)
    
    results = fcm.simulate_impact(model, impactor, h_start=100, craters=True, final_states=True,
                                  dedz=True, timeseries=True, seed=12)
    
    np.testing.assert_allclose(impactor.strength, results.bulk.timeseries["ram pressure"].iloc[-1])
    pd.testing.assert_series_equal(results.bulk.timeseries.iloc[-1],
                                   results.final_states.loc[results.bulk.id].iloc[1:],
                                   check_names=False)
    np.testing.assert_allclose(results.bulk.timeseries.index[-1].total_seconds(),
                               results.final_states.loc[results.bulk.id, "t"])
    
    all_ids = [results.bulk.id] + list(results.fragments.keys()) + list(results.clouds.keys())
    tmp = len(all_ids)
    all_ids = set(all_ids)
    assert len(all_ids) == tmp
    assert results.final_states.index.isin(all_ids).all()
    
    for ID, frag in results.fragments.items():
        pd.testing.assert_series_equal(frag.timeseries.iloc[-1],
                                       results.final_states.loc[ID].iloc[1:], check_names=False)
        np.testing.assert_allclose(frag.timeseries.index[-1].total_seconds(),
                                   results.final_states.loc[ID, "t"])
        
        assert frag.parents is not None and len(frag.parents) > 0
        if frag.children is not None:
            last_parent_state = frag.timeseries.iloc[-1]
            all_children_mass = 0
            for child_id in frag.children:
                child = results.fragments.get(child_id, results.clouds.get(child_id))
                assert child is not None
                assert child.parents[0] == ID
                first_child_state = child.timeseries.iloc[0]
                all_children_mass += first_child_state.m
                indices = ["x", "y", "z", "h"] # TODO: add "ram pressure"?
                pd.testing.assert_series_equal(first_child_state[indices], last_parent_state[indices])
            
            assert last_parent_state.m >= all_children_mass

    for ID, cloud in results.clouds.items():
        pd.testing.assert_series_equal(cloud.timeseries.iloc[-1],
                                       results.final_states.loc[ID].iloc[1:], check_names=False)
        np.testing.assert_allclose(cloud.timeseries.index[-1].total_seconds(),
                                   results.final_states.loc[ID, "t"])
        assert cloud.children is None
        
    
    assert results.bulk.parents is None
    assert results.bulk.children is not None and len(results.bulk.children) == groups[0].pieces
    last_bulk_state = results.bulk.timeseries.iloc[-1]
    all_children_mass = 0
    for child_id in results.bulk.children:
        child = results.fragments.get(child_id, results.clouds.get(child_id))
        assert child is not None
        assert child.parents is not None and len(child.parents) == 1
        assert child.parents[0] == results.bulk.id
        first_child_state = child.timeseries.iloc[0]
        all_children_mass += first_child_state.m
        indices = ["x", "y", "z", "h"] # TODO: add "ram pressure"?
        pd.testing.assert_series_equal(first_child_state[indices], last_bulk_state[indices])
        np.testing.assert_allclose(results.final_states.loc[child_id, "ram pressure"], groups[0].strength)
        
    np.testing.assert_allclose(last_bulk_state.m, all_children_mass)
    
    cratering_IDs = list()
    for IDs in results.craters.IDs:
        if isinstance(IDs, int):
            cratering_IDs.append(IDs)
        else:
            assert isinstance(cratering_IDs, (tuple, list))
            cratering_IDs += list(IDs)
    
    for ID in cratering_IDs:
        assert ID in results.final_states.index
        np.testing.assert_allclose(results.final_states.loc[ID, "z"], 0)
        np.testing.assert_allclose(results.final_states.loc[ID, "h"], 0)
    

if __name__ == "__main__":
    test_ablation_model()
    