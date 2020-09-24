# main class interface
__all__ = ["StructuralGroup", "FCMmeteoroid", "PancakeMeteoroid", "FragmentationMeteoroid",
           "FCMparameters", "simulate_impact"]

import math
from collections.abc import Iterable
import numpy as np
import pandas as pd

from . import _cpp_wrapper as cpp
from .models import *


###################################################
def _check_number(x, name, check_float=True, lower_bound=None, lower_incl=False,
                  upper_bound=None, upper_incl=False, bound_format="{:.0f}"):
    
    assert isinstance(name, str), "Bug: variable name must be a string"
    if check_float:
        if not isinstance(x, (int, float)):
            raise TypeError("{} must be a float".format(name))
    else:
        if not isinstance(x, int):
            raise TypeError("{} must be an int".format(name))
    
    if lower_bound is not None:
        assert isinstance(lower_bound, (int, float)), "Bug: lower bound must have type int or float"
        if lower_incl:
            if x < lower_bound:
                raise ValueError(("{} must be >= " + bound_format).format(name, lower_bound))
        elif x <= lower_bound:
            raise ValueError(("{} must be > " + bound_format).format(name, lower_bound))
    
    if upper_bound is not None:
        assert isinstance(upper_bound, (int, float)), "Bug: upper bound must have type int or float"
        if upper_incl:
            if x > upper_bound:
                raise ValueError(("{} must be <= " + bound_format).format(name, upper_bound))
        elif x >= upper_bound:
            raise ValueError(("{} must be < " + bound_format).format(name, upper_bound))
    
    if check_float:
        return float(x)
    return int(x)


###################################################
class StructuralGroup:
    """Structural group in the meteoroid
    
    See Wheeler et al. (2018) [https://doi.org/10.1016/j.icarus.2018.06.014]
    
    Parameters
    ----------
    mass_fraction : float
        Group mass as fraction of total meteoroid mass
        1 >= mass_fraction > 0
    
    density : float [kg/m^3]
        Material density of the group
        density > 0
    
    strength : float [kPa]
        Aerodynamic strength; if ram pressure exceeds this value, breakup is triggered
        strength > 0
    
    pieces : int, optional
        number of rubble pieces in this group
        pieces > 0
        default: 1
    
    cloud_mass_frac : float, optional
        Cloud mass fraction; fraction of the fragment mass that is released
            as a debris cloud on breakup
        0 <= cloud_mass_frac <= 1
        default: 0.5
    
    strength_scaler : float, optional
        Strength scaling parameter
        On breakup, aerodynamic strength of the fragments is increased according to
            an exponential Weibull-like scaling relation. See e.g. Wheeler et al. (2017)
            [https://doi.org/10.1016/j.icarus.2017.02.011]
        strength_scaler > 0
        default: 0.25 if cloud_mass_frac < 1, else None
    
    fragment_mass_fractions : tuple of float, optional
        On breakup, fragments with a fraction of (1 - cloud_mass_frac) * mass are generated
        Speficies how many and which fractions of this mass.
        all(0 < frac <= 1 for frac in fragment_mass_fractions)
        sum(fragment_mass_fractions) == 1
        default : (0.5, 0.5) if cloud_mass_frac < 1, else None
    """
    def __init__(self, mass_fraction, density, strength, pieces=1,  
                 cloud_mass_frac=0.5, strength_scaler=0.25,
                 fragment_mass_fractions=(0.5, 0.5)):
        
        self.mass_fraction = _check_number(mass_fraction, "mass_fraction", lower_bound=0,
                                           upper_bound=1, upper_incl=True)
        self.density = _check_number(density, "density", lower_bound=0)
        self.pieces = _check_number(pieces, "pieces", check_float=False, lower_bound=0)
        self.strength = _check_number(strength, "strength", lower_bound=0)
        self.cloud_mass_frac = _check_number(cloud_mass_frac, "cloud_mass_frac",
                                             lower_bound=0, lower_incl=True,
                                             upper_bound=1, upper_incl=True)
        self.strength_scaler = None
        self.fragment_mass_fractions = None
        
        if self.cloud_mass_frac < 1:
            self.strength_scaler = _check_number(strength_scaler, "strength_scaler", lower_bound=0)
            
            if not isinstance(fragment_mass_fractions, (list, tuple)):
                raise TypeError("fragment_mass_fractions must be a list or a tuple")
            if len(fragment_mass_fractions) == 0:
                raise ValueError("fragment_mass_fractions must not be empty")
                
            fragment_mass_fractions_sum = 0
            for f in fragment_mass_fractions:
                fragment_mass_fractions_sum += _check_number(f, "all f in fragment_mass_factions",
                                                             lower_bound=0, upper_bound=1,
                                                             upper_incl=True)
            if not math.isclose(fragment_mass_fractions_sum, 1):
                raise ValueError("sum of fragments_mass_fractions must equal 1, not {}".format(
                    fragment_mass_fractions_sum
                ))
            self.fragment_mass_fractions = tuple(float(i) for i in fragment_mass_fractions)
        else:
            assert math.isclose(self.cloud_mass_frac, 1), "this is a bug"


###################################################
class FCMmeteoroid:
    """Meteoroid with initial structure described in structural_groups
    
    Parameters
    ----------
    velocity : float [km/s]
        Velocity at atmospheric entry
        velocity >= 0
    
    angle : float [degrees]
        Angle of trajectory at atmospheric entry with respect to the ground
        0 = flies parallel to the ground; 90 = flies straight towards the ground
        0 < angle <= 90
    
    density : float [kg/m^3]
        Meteoroid material density
        density > 0
    
    radius : float [m]
        Meteoroid initial radius
        radius > 0
    
    strength : float [kPa]
        Aerodynamic strength; if ram pressure exceeds this value, breakup is triggered
        strength > 0
    
    cloud_mass_frac : float, optional
        Fraction of mass that turns into a debris cloud on initial breakup
        0 <= cloud_mass_frac <= 1
        default : 0
        
    structural_groups : list[StructuralGroup], optional
        Groups representing the internal structure of the meteoroid
        len(structural_groups) > 0 if cloud_mass_frac < 1
    """
    def __init__(self, velocity, angle, density, radius, strength,
                 cloud_mass_frac=0, structural_groups=None):
        
        self.velocity = _check_number(velocity, "velocity", lower_bound=0, lower_incl=True)
        self.angle = _check_number(angle, "angle", lower_bound=0, upper_bound=90, upper_incl=True)
        self.density = _check_number(density, "density", lower_bound=0)
        self.radius = _check_number(radius, "radius", lower_bound=0)
        self.strength = _check_number(strength, "strength", lower_bound=0)
        self.cloud_mass_frac = _check_number(cloud_mass_frac, "cloud_mass_frac", lower_bound=0,
                                             lower_incl=True, upper_bound=1, upper_incl=True)
        self.structural_groups = None
        if cloud_mass_frac < 1:
            self.structural_groups = list()
            if not isinstance(structural_groups, Iterable):
                raise TypeError("If cloud_mass_frac < 1, structural_groups must be an Iterable "\
                    + "of StructuralGroup objects")
            if len(structural_groups) == 0:
                raise ValueError("If cloud_mass_frac < 1, structural_groups must not be empty")
                
            for g in structural_groups:
                if isinstance(g, dict):
                    self.structural_groups.append(StructuralGroup(**g))
                else:
                    if not isinstance(g, StructuralGroup):
                        raise TypeError("all g in structural_groups StructuralGroup objects")
                    self.structural_groups.append(g)
            
            if min(g.strength for g in self.structural_groups) < self.strength:
                raise ValueError("group strength cannot be smaller than bulk strength")
            
            if not math.isclose(sum(g.mass_fraction for g in self.structural_groups), 1):
                raise ValueError("sum of group mass fractions must equal 1")
            
            if not sum(g.mass_fraction * self.mass / g.density
                       for g in self.structural_groups) <= self.volume * (1+1e-10):
                raise ValueError(
                    "Mass-weighted density of structural groups must be >= meteoroid density"
                )
    
    @property
    def volume(self):
        return 4*math.pi/3 * self.radius**3
    
    @property
    def mass(self):
        return self.density * self.volume
    
    def groups_table(self):
        if self.structural_groups is not None and len(self.structural_groups) > 0:
            table = pd.DataFrame([g.__dict__ for g in self.structural_groups])
            table['cloud_mass_frac'] *= 100
            table['component_mass'] = self.mass * table['mass_fraction'] / table['pieces']
            table['group_mass'] = table['component_mass'] * table.pieces
            table['mass_fraction'] *= 100
            
            table.index.name = 'Group ID'
            table.columns = table.columns.map({
                'pieces': "Number of pieces in group",
                'mass_fraction': "Group mass fraction (%)",
                'density': "Group density (kg/m^3)",
                'radius': "Component radii (m)",
                'strength': "Strength (kPa)",
                'strength_scaler': "Strength scaling exponent",
                'cloud_mass_frac': "Cloud mass fraction (%)",
                'fragment_mass_fractions': "Fragment mass fractions",
                'component_mass': "Component masses (kg)",
                'group_mass': "Group mass (kg)"
            })
            assert not table.columns.isnull().any(), "Bug: unnamed column"
        else:
            table = None
        
        return table


###################################################
class PancakeMeteoroid(FCMmeteoroid):
    """Restriction of the FCMmeteoroid to the could-only models (pancake, debris cloud, chain reaction)
    
    Parameters
    ----------
    velocity : float [km/s]
        Velocity at atmospheric entry
        velocity >= 0
    
    angle : float [degrees]
        Angle of trajectory at atmospheric entry with respect to the ground
        0 = flies parallel to the ground; 90 = flies straight towards the ground
        0 < angle <= 90
        
    density : float [kg/m^3]
        Meteoroid material density
        density > 0
    
    radius : float [m]
        Radius of individual piece in group
        radius > 0
    
    strength : float [kPa]
        Aerodynamic strength; if ram pressure exceeds this value, breakup is triggered
        strength > 0
    """
    def __init__(self, velocity, angle, density, radius, strength):
        super().__init__(velocity, angle, density, radius, strength, 1)
        

###################################################
class FragmentationMeteoroid(FCMmeteoroid):
    """Restriction of the FCMmeteoroid to fragmentation-only models
    
    Parameters
    ----------
    velocity : float [km/s]
        Velocity at atmospheric entry
        velocity >= 0
    
    angle : float [degrees]
        Angle of trajectory at atmospheric entry with respect to the ground
        0 = flies parallel to the ground; 90 = flies straight towards the ground
        0 < angle <= 90
        
    density : float [kg/m^3]
        Meteoroid material density
        density > 0
    
    radius : float [m]
        Radius of individual piece in group
        radius > 0
    
    strength : float [kPa]
        Aerodynamic strength; if ram pressure exceeds this value, breakup is triggered
        strength > 0
    
    strength_scaler : float, optional
        Strength scaling parameter
        On breakup, aerodynamic strength of the fragments is increased according to
            an exponential Weibull-like scaling relation. See Wheeler et al. (2017)
            [https://doi.org/10.1016/j.icarus.2017.02.011]
        strength_scaler > 0
        default: 0.25
    
    fragment_mass_fractions : tuple of float, optional
        On breakup, fragments with a fraction of (1-cloud_mass_frac)*mass are generated
        Speficies how many and which fractions of this mass.
        default : (0.5, 0.5)
    """
    def __init__(self, velocity, angle, density, radius, strength,
                 strength_scaler=0.25, fragment_mass_fractions=(0.5, 0.5)):
        group = StructuralGroup(1, density, strength, cloud_mass_frac=0,
                                strength_scaler=strength_scaler,
                                fragment_mass_fractions=fragment_mass_fractions)
        super().__init__(velocity, angle, density, radius, strength, structural_groups=[group])


###################################################
def default_cratering_params(ground_type):
    """ Sources: - Daubar et al. (2020), p.? [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JE006382]
    - Master's thesis by Eric
    """
    gt = GroundType(ground_type)
    if gt is GroundType.regolith:
        return CrateringParams(ground_density=1.5e3, ground_strength=1e1, K1=0.15, K2=1.0, Kr=1.1,
                               mu=0.4, nu=1/3, rim_factor=1.3)
        
    if gt is GroundType.hard_soil:
        return CrateringParams(ground_density=2.1e3, ground_strength=1.3e2, K1=0.04, K2=1.0, Kr=1.1,
                               mu=0.55, nu=1/3, rim_factor=1.3)
    
    if gt is GroundType.cohesionless_material:
        return CrateringParams(ground_density=1.5e3, ground_strength=0, rim_factor=1.3, K1=0.133,
                               K2=0, mu=0.41, nu=0.4, Kr=1.25)
    
    if gt is GroundType.cohesive_soil:
        return CrateringParams(ground_density=1.5e3, ground_strength=50, rim_factor=1.3, K1=0.133,
                               K2=1, mu=0.41, nu=0.4, Kr=1.25)
    
    raise NotImplementedError(
        "Default parameters for ground type '{}' not specified".format(gt.value)
    )


###################################################
def default_cloud_dispersion_coeff(cloud_disp_model):
    """Default values for coefficients by different cloud dispersion models
    
    - Pancake model 'PM' (Chyba et al, 1993): 1.5; equal to drag coefficient
    - Debris cloud model 'DCM' (Hills and Goda, 1993): 0.6; proposed by Wheeler et al. (2018)
    - Chain reaction model 'CRM' (Avramenko et al, 2014): 
    """
    model = CloudDispersionModel(cloud_disp_model)
    if model is CloudDispersionModel.CRM:
        return 1.5
    if model is CloudDispersionModel.DCM:
        return 0.6
    if model is CloudDispersionModel.PM:
        return 1.5
    
    raise NotImplementedError(
        "Default coefficient for cloud dispersion model '{}' not specified".format(model.value)
    )


###################################################
class FCMparameters:
    """Fragment-Cloud Model Parameters
    
    For model description, see Wheeler et al. (2018) [https://doi.org/10.1016/j.icarus.2018.06.014]
    
    Parameters
    ----------    
    g0 : float [m/s^2]
        Gravitational acceleration at height 0
        g0 >= 0
    
    Rp : float [km] or math.inf
        Planet radius; planet is assumed to be spherical
        If Rp = np.inf, g = g0 at all heights
        Rp > 0
    
    atmospheric_density : array-like or pandas.Series
        2d array [height [km], density [kg/m^3]]
        density > 0; height in ascending order; density monotonically decreasing
        will be interpolated exponentially; above largest height, density is assumed to be 0
        
    ablation_coeff : float [kg/J], optional
        Ablation coefficient; meteoroid mass loss due to friction in the atmoshere
        ablation_coeff >= 0
        default : 5e-9 ; see Wheeler et al. (2018)
    
    drag_coeff : float, optional
        Drag coefficient between meteoroid material and atmosphere
        drag_coeff >= 0
        default : 1.5
    
    lift_coeff : float, optional
        Lift coefficient of meteoroid in the atmosphere
        lift_coeff >= 0
        default : 5e-4
        
    max_strength : float [kPa], optional
        Maximum aerodynamic strength for a fragment
        max_strength > 0
        default : 3.3e5 ; see Popova et al. (2013) [https://doi.org/10.1126/science.1242642]
    
    frag_model : str, optional
        Fragmentation model; one of FragmentationModel enum
        default : 'IW' (used by Wheeler et al. (2018))
    
    frag_velocity_coeff : float, optional
        Bow shock interaction coefficient at fragmentation in case of 'IW' model
        Fragments separate with a velocity perpendicular to trajectory (Passey and Melosh, 1980)
            [https://doi.org/10.1016/0019-1035(80)90072-X]
        frag_velocity_coeff > 0
        default : 0.19/1.5 ; see Artemieva and Shuvalov (2001)
            [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2000JE001264]
    
    cloud_disp_model : str, optional
        Cloud dispersion model; one of CloudDispersionModel enum
        default : 'DCM'
        
    cloud_disp_coeff : float, optional
        Cloud dispersion coefficient
        cloud_disp_coeff > 0
        default : default_cloud_dispersion_coeff(cloud_disp_model)
        
    strengh_scaling_disp:
        At breakup, strength is increased according to Weibull-like scaling relation
            Artemieva and Shuvalov (2001) introduced that actual new strength is drawn from
            random normal distribution with sigma = strengh_scaling_disp * mean
            [https://doi.org/10.1029/2000JE001264]
        strengh_scaling_disp >= 0
        default: 0.5
    
    fragment_mass_disp : float, optional
        How much to randomly vary mass fractions of fragments.
        0 = no variation, 0.5 = mass of fragment varies from 50% to 100% of chosen fraction
        0 <= fragment_mass_disp < 1
        default : 0.9 ; see Newland (2019)
    
    min_crater_radius : float, optional
        Minimum detectable crater radius, in [m].
            Fragments that would produce a smaller crater are dropped from the simulation
            to increase performance.
        0 <= min_crater_radius
        default : 0.5 ; see Daubar et al. (2019)
            [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018JE005857]
    
    cratering_params : CrateringParams, optional
        Parameters for calculating the impact crater diameter
        default : default_cratering_params("cohesive_soil")
    
    dh : float [m], optional
        height resolution of dEdz
        dh >= 0.1
        Default : 10
    
    variable_timestep : bool, optional
        Whether to use variable time step size integration with target :param: precision, or
        fixed step size with size = :param: precision
        Default: True
        
    precision : float, optional
        If variable_timestep = True: Precision target for variable time step
            Else: fixed timestep in [s]
        1e-7 < precision <= 1
        Default : 1e-4
    
    timestepper : str, optional
        Time stepping scheme used for ODE solving; one of Timestepper enum
        Default : 'AB2'
    """
    def __init__(self, g0, Rp, atmospheric_density,
                 ablation_coeff=5e-9, drag_coeff=1.5, lift_coeff=5e-4, max_strength=3.3e5,
                 frag_model="IW", frag_velocity_coeff=0.19/1.5, cloud_disp_model="DCM",
                 cloud_disp_coeff=None, strengh_scaling_disp=0.5, fragment_mass_disp=0.9,
                 min_crater_radius=0.5, cratering_params="cohesive_soil",
                 dh=10, variable_timestep=True, precision=1e-4, timestepper="AB2"):
        
        self.g0 = _check_number(g0, "g0", lower_bound=0)
        self.Rp = _check_number(Rp, "Rp", lower_bound=0)
        
        if isinstance(atmospheric_density, (list, tuple, np.ndarray)):
            self.atmospheric_density = np.array(atmospheric_density).astype(np.float64)
        elif isinstance(atmospheric_density, (pd.Series, pd.DataFrame)):
            self.atmospheric_density = np.array([atmospheric_density.index.to_numpy(),
                                                 atmospheric_density.to_numpy()]).astype(np.float64)
        else:
            raise TypeError("atmospheric_density must be array-like")
        
        if not np.isfinite(self.atmospheric_density).all():
            raise ValueError("atmospheric_density must not contain NaN for inf values")
        if not self.atmospheric_density.ndim == 2:
            raise ValueError("atmospheric_density must be a two-dimensional array")
        if not (self.atmospheric_density[1] > 0).all():
            raise ValueError("all atmospheric_density values must be > 0")
        if not (np.diff(self.atmospheric_density[0]) > 0).all():
            raise ValueError("height in atmospheric_density must be in ascending order")
        if not (np.diff(self.atmospheric_density[1]) < 0).all():
            raise ValueError("atmospheric_density must decrease monotonically as height increases")
        
        self.ablation_coeff = _check_number(ablation_coeff, "ablation_coeff",
                                            lower_bound=0, lower_incl=True)
        self.drag_coeff = _check_number(drag_coeff, "drag_coeff", lower_bound=0, lower_incl=True)
        self.lift_coeff = _check_number(lift_coeff, "lift_coeff", lower_bound=0, lower_incl=True)
        self.max_strength = _check_number(max_strength, "max_strength", lower_bound=0)
        self.frag_model = FragmentationModel(frag_model)
        self.frag_velocity_coeff = _check_number(frag_velocity_coeff, "frag_velocity_coeff",
                                                 lower_bound=0)
        self.cloud_disp_model = CloudDispersionModel(cloud_disp_model)
        
        if cloud_disp_coeff is None:
            self.cloud_disp_coeff = default_cloud_dispersion_coeff(self.cloud_disp_model)
        else:
            self.cloud_disp_coeff = _check_number(cloud_disp_coeff, "cloud_disp_coeff",
                                                  lower_bound=0)
            
        self.strengh_scaling_disp = _check_number(strengh_scaling_disp, "strengh_scaling_disp",
                                                  lower_bound=0, lower_incl=True)
        self.fragment_mass_disp = _check_number(fragment_mass_disp, "fragment_mass_disp",
                                                lower_bound=0, lower_incl=True, upper_bound=1)
        self.min_crater_radius = _check_number(min_crater_radius, "min_crater_radius",
                                               lower_bound=0, lower_incl=True)
        
        if isinstance(cratering_params, dict):
            self.cratering_params = CrateringParams(**cratering_params)
        elif isinstance(cratering_params, (str, GroundType)):
            self.cratering_params = default_cratering_params(cratering_params)
        elif isinstance(cratering_params, CrateringParams):
            self.cratering_params = cratering_params
        else:
            raise TypeError("cratering_params must be a CrateringParams object")
        
        self.dh = _check_number(dh, "dh", lower_bound=0.1, lower_incl=True, upper_bound=1000,
                                upper_incl=True, bound_format="{:.0E}")
        self.precision = _check_number(precision, "precision", lower_bound=1e-7, lower_incl=True,
                                       upper_bound=0.1, upper_incl=True, bound_format="{:.0E}")
        if not isinstance(variable_timestep, bool):
            raise TypeError("variable_timestep must be a bool")
        self.variable_timestep = bool(variable_timestep)
        self.timestepper = Timestepper(timestepper)


###################################################        
def simulate_impact(params, meteoroid, h_start,
                    h_ground=None, craters=True, final_states=False,
                    dedz=True, timeseries=False, seed=0):
    """Solve the atmospheric entry
    
    Parameters
    ----------
    params : FCMparameters
        FCM model parameters and settings
    
    meteoroid : FCMmeteoroid
        impactor state right before it contacts the planetary atmosphere
    
    h_start : float
        altitude in [km] above standard 0 where simulation starts

    h_ground : float, optional
        altitude in [km] of planetary surface, relative to standard 0
        default : min(params.atmospheric_density[0])
    
    craters : bool, optional
        Whether to calculate and return crater cluster
        default : True
    
    final_states : bool, optional
        Whether to return the final state (mass, velocity etc.) of all fragments
        default : False
    
    dedz : bool, optional
        Whether to calculate and return the sum of the energy deposition
        default : True
        
    timeseries : bool, optional
        Whether to save and return all time series of all fragments
        default : False
    
    seed : int, optional
        seed for the random number generator
        seed >= 0
        default : 0
    
    Returns
    -------
    FcmResults

    """
    if not isinstance(params, FCMparameters):
        raise TypeError("params must be a FCMparameters object")
    if not isinstance(meteoroid, FCMmeteoroid):
        raise TypeError("meteoroid must be a FCMmeteoroid object")
    
    h_start = _check_number(h_start, "h_start")
    if h_ground is None:
        h_ground = params.atmospheric_density[0].min()
    else:
        h_ground = _check_number(h_ground, "h_ground")
    if not h_start > h_ground:
        raise ValueError("start height must be larger than ground height")
    
    if meteoroid.structural_groups is not None:
        if any(g.strength > params.max_strength for g in meteoroid.structural_groups):
            raise ValueError("group strength must not be larger than params.max_strength")
    if meteoroid.strength > params.max_strength:
        raise ValueError("meteoroid bulk strength must not be larger than params.max_strength")
    
    seed = _check_number(seed, "seed", check_float=False, lower_bound=0, lower_incl=True,
                         upper_bound=cpp.max_seed())
    
    craters, energy_deposition, final_states_frame, bulk, fragments, clouds = cpp.run_simulation(
        params, meteoroid, h_start, h_ground, craters, final_states, dedz, timeseries, seed
    )
    
    return FcmResults(craters=craters, final_states=final_states_frame,
                      energy_deposition=energy_deposition, bulk=bulk, fragments=fragments,
                      clouds=clouds)
