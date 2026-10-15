''' calc.py
    created by Sami Turbeville
    on 9/21/23

    Some basic functions for convert or calculate
    atmospheric variables. Makes extensive use of 
    metpy units and xarray. 
'''
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from geopy.distance import geodesic
from metpy.units import units
import metpy.constants as mpconstants


# universal constants
R = mpconstants.dry_air_gas_constant
g = mpconstants.earth_gravity
Rearth = mpconstants.Re


def omega2w(omega, p, t):
    ''' converts vertical velocity from pressure coords
        to m/s.

        input:
         - omega : array in units of Pa/s
         - p     : pressure array in Pa
         - t     : temperature array in K
        output:
         - w     : vertical velocity array in units of m/s
    '''
    # print(p.to_base_units(), "Pa")
    rho = p.to_base_units() / (R*t.metpy.convert_units('K'))
    w = (-omega.metpy.convert_units('Pa/s')/(rho*g)).metpy.convert_units('m/s')
    return w


def w2omega(w, p, t):
    ''' converts vertical velocity from m/s
        to pressure coords (Pa/s).

        input:
         - w     : array in units of m/s
         - p     : pressure array in Pa
         - t     : temperature array in K
        output:
         - omega : vertical velocity array in units of Pa/s
    '''
    rho = p.metpy.convert_units('Pa') / (R*t.metpy.convert_units('K'))
    omega = (-w.metpy.convert_units('m/s')*rho*g).metpy.convert_units('Pa/s')
    return omega

def geopot2height(phi):
    """
    Converts geopotential (m2/s2) to height (m). 

    $ z = \frac{\Phi R_e}{gR_e - \Phi} $
    where R_e is the radius of Earth, g is gravitational constant, and
    
    :param phi: geopotential height in m2/s2
    """
    return (phi * Rearth) / (g * Rearth - phi)

def calc_rice(qi, ni):
    rho = 920.0 * units('kg/m^3')# kg/m3
    qi = qi.metpy.convert_units('kg/kg')
    ni = ni.metpy.convert_units('1/kg')
    # r_ice = np.where((ni>1e-5),(3*qi/(4*np.pi*rho*ni))**(1/3),0)*1e6  # um
    r_ice = np.cbrt(3*qi/(4*np.pi*rho*ni))
    r_ice = r_ice.metpy.convert_units('micrometers')  # um
    return r_ice

def calc_ni(numice, qv, p, t, debug=True):
    """returns icnc as 1/cm3"""
    if debug:
        print("UNIT CHECK:")
        print("\tNUMICE:", numice.metpy.units)
        print("\tQV    :", qv.metpy.units)
        print("\tPRES  :", p.metpy.units)
        print("\tTEMP  :", t.metpy.units)
    ni = numice * calc_rho(qv,p,t)  # 1/kg * kg/m3 = 1/m3
    return ni.metpy.convert_units('cm^-3')

def calc_rho(qv, p, t):
    """calculates density of air in kg/m3 for given input"""
    Tv = (1 + 0.61*qv) * t  # K
    rho = p / (R*Tv)  # kg/m3
    return rho

def calc_theta(t):
    """calculates the potential temperature, theta, of air
       from the temperature xarray
    """
    theta = t * ((1000/t.lev)**(0.286))
    return theta

def calc_rhice(ds, varQ="Q", varT="T", z_units="hPa"):
    """ input: xarray with variables Q and T
        output: xarray of rh wrt ice
    """
    e_si = np.exp(9.550426 - 5723.265/ds[varT] +
                  3.53068*np.log(ds[varT]) - 0.00728332*ds[varT])
    if z_units=="hPa":
        z = ds.lev*100
    else:
        z=ds.lev
    w_si = (0.622 * e_si) / (z - e_si)
    w_i = ds[varQ] / (1 - ds[varQ])
    rh_ice = w_i/w_si * 100
    return rh_ice

def fix_longitudes(lon):
    """fix longitude so that it is always from -180 to 180 
       (not 0 to 360)"""
    return ((lon + 180) % 360) - 180

def calc_geodesic_distance(latlon0, latlon1):
    """ Calculates the distance between two points
        given in (lon) coords, returns the distance
        in kilometers
        Uses metpy package
    """
    return( geodesic(latlon0, latlon1).km * units.km)
