__all__ = ["AblationModel", "Fragment", "Timestepper", "CloudDispersionModel", "FragmentationModel",
           "GroundType", "CrateringParams", "FcmResults"]

from enum import Enum
from collections import namedtuple


###################################################
Fragment = namedtuple(
    "Fragment",
    ['timeseries',  # (pd.DataFrame) index = t; columns = [x, y, z, h, v, r, angle, m, dedz]
                    #    where  t =     time [s] since atmospheric entry
                    #           x =     center x position [km],
                    #           y =     center y position [km],
                    #           z =     center height above sea level [km],
                    #           v =     velocity [km/s],
                    #           angle = tranjectory angle [deg] w.r.t. the ground (0 = horizontal, 90=vertical),
                    #           r =     fragment radius [m] (perpendicular to trajectory),
                    #           m =     fragment mass [kg],
                    #           dedz =  energy deposited in atmosphere [kt TNT/km]
                    #           ram pressure [kPa]
     'strength',    # (float)               aerodynamic strength [kPa] of the fragment material
     'id',          # (int)                 fragment ID
     'parents',     # tuple(int) or None    IDs of parents from which fragment separated, if fragment is not the bulk
     'children'],   # list(int) or None     list of child fragments if fragmentation event occurred
    
    defaults=[0,     # id
              None,  # parents
              None]  # children
)


###################################################
class Timestepper(Enum):
    forward_euler  = 'forward_euler'
    improved_euler = 'improved_euler'
    RK4            = 'RK4'      # 4th order explicit Runge-Kutta scheme
    AB2            = 'AB2'      # 2nd order explicit multi-step Adams-Bashforth scheme


###################################################
class AblationModel(Enum):
    chain_reaction = 'chain_reaction'  # part of the Chain Reaction Model before meteoroid breakup,
                                       # which keeps the meteoroid's shape spherical
                                       #   by Avramenko et al. (2014) [https://doi.org/10.1002/2013JD021028]

    constant_r = 'constant_r'          # radius perpendicular to trajectory remains constant, 
                                       # as described in the 'pancake' and 'debris cloud' models
                                       #  by Chyba et al. (1993) [https://doi.org/10.1038/361040a0]
                                       #  and by Hills and Goda (1993) [https://doi.org/10.1086/116499]

    # TODO: add comet model by Crawford, 1996 [https://doi.org/10.1017/S0252921100115490] to the numerical solver


###################################################
class CloudDispersionModel(Enum):
    PM  = "PM"      # Pancake Model
                    #   by Chyba et al. (1993) [https://doi.org/10.1038/361040a0]
                    
    DCM = "DCM"     # Debris Cloud Model
                    #   by Hills and Goda (1993) [https://doi.org/10.1086/116499]
                    
    CRM = "CRM"     # Chain Reaction Model
                    #   by Avramenko et al. (2014) [https://doi.org/10.1002/2013JD021028]


###################################################
class FragmentationModel(Enum):
    IW = "IW"   # Independent wakes model
                #   by Passey and Melosh (1980) [https://doi.org/10.1016/0019-1035(80)90072-X]
                
    CW = "CW"   # Collective wake model
                #   by ReVelle (2006) [https://doi.org/10.1017/S1743921307003122]
                #   not implemented
                
    NCW = "NCW" # Non-collective wake model
                #   also by ReVelle (2006)
                #   not implemented


###################################################
class GroundType(Enum):
    regolith = "regolith"
    hard_soil = "hard_soil"
    cohesionless_material = "cohesionless_material"
    cohesive_soil = "cohesive_soil"
    dry_soil = "dry_soil"


###################################################
CrateringParams = namedtuple(
    "CrateringParams",
    ['ground_density',              # Density of ground material (kg/m^3)
     'ground_strength',             # Cohesive strenght of the ground material (kPa)
     'rim_factor',                  # scaling factor for calculating rim-to-rim impact crater diameter
     'K1', 'K2', 'Kr', 'mu', 'nu']  # coefficients in Holsapple crater radius equation
)


###################################################
FcmResults = namedtuple(
    "FcmResults",
    ['craters',             # pandas.DataFrame(x, y, r) : x-pos in [m], y-pos in [m], radius in [m]
                            #       representation of crater cluster
     'energy_deposition',   # pandas.Series(index = altitiude [km], data = dEdz in kt TNT / km)
                            #       sum of energy deposition of all fragments
     'final_states',        # pandas.DataFrame(index = int(ID), columns = same as fcm.Fragment.timeseries)
                            #       collection of states of all fragments just before
                            #       a) impact, b) atmoshperic escape, c) break up, d) total ablation
     'bulk',                # fcm.Fragment : time series of initial bulk
     'fragments',           # dict(int(ID): fcm.Fragment) : time series of all fragments
     'clouds']              # dict(int(ID): fcm.Fragment) : time series of all debris clouds
)


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
