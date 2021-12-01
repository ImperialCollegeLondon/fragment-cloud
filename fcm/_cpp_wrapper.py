"""wraps the C++ functions"""

import math
import numpy as np
import pandas as pd

from . import core
from .models import *  


###################################################
ts_map = {Timestepper.forward_euler: core.OdeSolver.forward_euler,
          Timestepper.improved_euler: core.OdeSolver.improved_euler,
          Timestepper.RK4: core.OdeSolver.RK4,
          Timestepper.AB2: core.OdeSolver.AB2}

cloud_map = {CloudDispersionModel.PM: core.CloudDispersionModel.pancake,
             CloudDispersionModel.DCM: core.CloudDispersionModel.debris_cloud,
             CloudDispersionModel.CRM: core.CloudDispersionModel.chain_reaction}

abl_map = {AblationModel.chain_reaction: core.AblationModel.meteoroid,
           AblationModel.constant_r: core.AblationModel.meteoroid_const_r}


###################################################
def _core_params(model, timeseries):
    
    if model.Rp == np.inf:
        Rp = 1000.0
        use_flat_planet = True
    else:
        Rp = model.Rp * 1e3
        use_flat_planet = False
    
    core_crater_coeff = core.FcmCraterCoeff(
        model.min_crater_radius, model.cratering_params.ground_density,
        model.cratering_params.ground_strength*1e3, model.cratering_params.K1,
        model.cratering_params.K2, model.cratering_params.Kr, model.cratering_params.mu,
        model.cratering_params.nu, model.cratering_params.rim_factor
    )
    
    core_params = core.FcmParams(
        model.g0, Rp, model.ablation_coeff, model.drag_coeff, model.lift_coeff,
        model.max_strength*1e3, model.frag_velocity_coeff, model.cloud_disp_coeff,
        model.strengh_scaling_disp, model.fragment_mass_disp, core_crater_coeff
    )
    
    core_settings = core.FcmSettings(cloud_map[model.cloud_disp_model], ts_map[model.timestepper],
                                     model.precision, model.dh, timeseries, use_flat_planet,
                                     not model.variable_timestep, abl_map[model.ablation_model])
    
    return core_params, core_settings


###################################################
def _core_structural_group(group):
    
    if group.fragment_mass_fractions is None:
        np.testing.assert_allclose(group.cloud_mass_frac, 1)
        assert group.strength_scaler is None or np.isnan(group.strength_scaler)
        fragment_mass_fractions_list = list()
        strength_scaler = 1
    else:
        assert group.cloud_mass_frac < 1
        assert group.strength_scaler is not None
        strength_scaler = group.strength_scaler
        fragment_mass_fractions_list = list(group.fragment_mass_fractions)
    
    result = core.StructuralGroup(group.mass_fraction, group.pieces, group.strength*1e3,
                                  group.density, group.cloud_mass_frac, strength_scaler,
                                  fragment_mass_fractions_list)
    return result


###################################################
def _core_meteoroid(meteoroid):
    
    if meteoroid.structural_groups is not None:
        groups = [_core_structural_group(g) for g in meteoroid.structural_groups]
    else:
        groups = list()
    
    core_m = core.Meteoroid(meteoroid.density, meteoroid.velocity*1e3, meteoroid.radius,
                            np.pi * meteoroid.angle / 180, meteoroid.strength * 1e3,
                            meteoroid.cloud_mass_frac, groups)
    np.testing.assert_allclose(core_m.mass, meteoroid.mass)
    
    return core_m


###################################################
def _make_dataframe(data, index, h_ground, time_column=False):
    columns = ['m', 'v', 'angle', 'z', 'x', 'y', 'r', 'density', 'dEdz', 'ram pressure']
    if time_column:
        columns = ['t'] + columns
    
    ts = pd.DataFrame(columns=columns, index=index, data=data)
    ts['angle'] *= 180/np.pi            # radians -> degrees
    ts[['x', 'y', 'z', 'v']] *= 1e-3    # m -> km
    ts['h'] = ts.z - h_ground
    ts['dEdz'] /= 4.184e9               # J/m -> kt TNT / km
    ts['ram pressure'] *= 1e-3          # Pa -> kPa
    
    return ts
    

###################################################
def run_simulation(model, meteoroid, h_start, h_ground, craters=True, final_states=False, dedz=True,
                   timeseries=False, seed=0):

    params, settings = _core_params(model, timeseries)
    core_m = _core_meteoroid(meteoroid)
    
    craters_tuple, dEdz_data, final_states_tuple, data = core.solve_impact(
        core_m, h_start*1e3, h_ground*1e3, params, settings, model.atmospheric_density[0] * 1e3,
        model.atmospheric_density[1], int(seed), craters, dedz, final_states
    )
    
    if craters and craters_tuple[0].size > 0:
        assert craters is not None
        weights = craters_tuple[2]**3
        x = craters_tuple[0] - np.average(craters_tuple[0], weights=weights)
        y = craters_tuple[1] - np.average(craters_tuple[1], weights=weights)
        
        craters = pd.DataFrame({"x": x, "y": y, "r": craters_tuple[2]})
        if len(craters_tuple[3]) > 0:
            craters["IDs"] = craters_tuple[3]
    else:
        craters = None
    
    if dedz:
        assert dEdz_data is not None
        dEdz_index = np.linspace(h_start, h_ground, math.ceil(1e3*(h_start - h_ground) / model.dh) + 1)
        dEdz_series = pd.Series(index=dEdz_index, data=dEdz_data / 4.184e9)
    else:
        dEdz_series = None
    
    if final_states:
        assert final_states_tuple is not None
        final_states_frame = _make_dataframe(final_states_tuple[1],
                                             pd.UInt64Index(final_states_tuple[0]), h_ground,
                                             True)
    else:
        final_states_frame = None
    
    bulk = None
    fragments = dict()
    clouds = dict()
    
    if timeseries:
        assert len(data) > 0
        for i, (info, array) in enumerate(data):
            ID = info['ID']
            ts = _make_dataframe(array[:, 1:], pd.TimedeltaIndex(array[:, 0], unit='s', name=ID),
                                 h_ground)
            fragment = Fragment(
                timeseries=ts, strength=info['strength']*1e-3, id=ID,
                parents=info['parent IDs'] if len(info['parent IDs']) > 0 else None,
                children=info['daughter IDs'] if len(info['daughter IDs']) > 0 else None
            )
            if i == 0:
                bulk = fragment
            elif info['is cloud']:
                assert ID not in clouds, "Bug: duplicate ID"
                clouds[ID] = fragment
            else:
                assert ID not in fragments, "Bug: duplicate ID"
                fragments[ID] = fragment
        
    return craters, dEdz_series, final_states_frame, bulk, fragments, clouds


###################################################
def max_seed():
    return core.max_seed()
