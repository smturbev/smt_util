"""
spectra_util.py

code of vertical velocity spectra analysis
originally from Rachel Atlas (see Fig. 8 in Atlas & Bretherton, 2023)
adapted by Sami Turbeville for SCREAM @smturbev
"""

import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import griddata
import metpy.calc as mpcalc
from metpy.units import units
from geopy.distance import geodesic

from . import calc


def regrid_w(file_name, lev=(250*units.hPa), method="nearest",
             omega_var_name="omega", t_var_name="T_mid",
             lat_var=None, lon_var=None, w_var=None, 
             lat_res=None, lon_res=None):
    """
    Interpolates unstructured data in 2 spatial dimensions and over time
        at a specific vertical level.

    Parameters
    ----------
    file_name : str
        Path to file with omega/w variables
    lat_bnds : tuple of ints with metpy units
        Lower and upper bounds on the y direction or latitude
    lon_bnds : tuple of ints with metpy units
        Lower and upper bounds on the x direction or longitude
    lev : metpy.Quantity
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for scipy.interpolate.griddata()
    lat_var : str, optional
        Name of variable for y direction: default = 'lat'
    lon_var : str, optional
        Name of variable for x direction: default = 'lon'
    w_var : str, optional
        Name of variable for vertical velocity: default = 'W'
    lat_res: float metpy.Quantity, optional
        Distance between grid points in the y direction: default = 3.3 km
    lon_res: float metpy.Quantity, optional
        Distance between grid points in the x direction: default = 3.3 km

    Returns
    -------
    w_regridded : xarray.DataArray with dimentions of (time, lat, lon)
        Xarray of regridded vertical velocity xarray.DataArray.
    
    """
    # initialize optional parameters
    if lat_var is None:
        lat_var='lat'
    if lon_var is None:
        lon_var='lon'
    if w_var is None:
        w_var='W'
    if lat_res is None:    
        lat_res=3.25 * units.km
    if lon_res is None:
        lon_res = lat_res
    print("RESOLUTION", lat_res, lon_res)
    # lat0, lat1 = lat_bnds
    # lon0, lon1 = lon_bnds
    
    unstruc_lat = xr.open_dataset(file_name)[lat_var].metpy.quantify() 
    unstruc_lon = xr.open_dataset(file_name)[lon_var].metpy.quantify() 
    unstruc_lat = unstruc_lat.metpy.unit_array
    unstruc_lon = unstruc_lon.metpy.unit_array
    
    lat0 = np.nanmin(unstruc_lat)
    lat1 = np.nanmax(unstruc_lat)
    lon0 = np.nanmin(unstruc_lon)
    lon1 = np.nanmax(unstruc_lon)

    print(lat0)
    
    try: # units are in meters or the like
        dlon = abs(lon1 - lon0)
        dlat = abs(lat1 - lat0)
        dlat.to('km')
        print("units in meters")
    except: # units are in degrees_north or degrees_east
        # lat0, lat1, lon0, lon1 = lat0.m, lat1.m, lon0.m, lon1.m
        dlon = calc_geodesic_distance((lat0,lon0),(lat0,lon1))
        dlat = calc_geodesic_distance((lat0,lon0),(lat1,lon0))
        print("units in degrees latlon")
    nlon = int(dlon//lon_res)
    nlat = int(dlat//lat_res)
    print("nlon/nlat",nlon,nlat)

    # only use square regions to make sure the x-y length scales are the same
    # so that we see the same range of wavelengths for each dimension
    # make sure that the region of analysis is a square
    print(f"The distance in the x direction, {lon_var}, is {dlon.to('km')} with grid spacing of {lon_res}.")
    print(f"The distance in the y direction, {lat_var}, is {dlat.to('km')} with grid spacing of {lat_res}.")
    
    # Interpolate to a grid much finer than the real resolution
    # to capture all variability
    # near native regular lat lon grid but in km
    nn_reg_lat = np.linspace(0,dlat,nlat+1)
    nn_reg_lon = np.linspace(0,dlon,nlon+1)
    X,Y = np.meshgrid(nn_reg_lat, nn_reg_lon)
    print("X,Y shapes",X.shape)
    
    # check that the grid is a square
    # assert(nn_reg_lat.shape == nn_reg_lat.shape)
    w_ds = open_w_dataarray(file_name, omega_var_name, t_var_name)[w_var]
    if 'lev' in w_ds.dims:
        w_ds = w_ds.sel(lev=lev, method="nearest")

    # loop through all timesteps 
    nt = w_ds.time.shape[0]
    w_gridded = np.zeros((nt, nlon+1, nlat+1))
    print(w_gridded.shape)
    for t in range(nt):
        w_gridded[t,:,:] = griddata((unstruc_lon, unstruc_lat),
                                    w_ds.isel(time=t),
                                    (X,Y),
                                    method=method)
    w_gridded = xr.DataArray(w_gridded,
                             dims=['time','lon','lat'],
                             coords={'time':w_ds.time,
                                     'lon':nn_reg_lon,
                                     'lat':nn_reg_lat},
                            attrs={"long_name": "vertical velocity regridded to near native regulat lat-lon grid",
                                   "standard_name":w_var,
                                   "units":"m/s"})
    return w_gridded


def calc_psd_welch(w_latlon, t0=4, fs=1.0, **kwargs):
    """
    Estimate power spectral density using Welch's method from scipy.signal.welch().

    Welch's method [1]_ computes an estimate of the power spectral
    density by dividing the data into overlapping segments, computing a
    modified periodogram for each segment and averaging the
    periodograms.

    Parameters
    ----------
    w_latlon : ndarray with dimensions (time, lon, lat)
        xarray.DataArray of vertical velocity on a regular lat-lon grid
    fs : float, optional
        Sampling frequency of the xy grid - ie. the grid spacing. Defaults to 1.0.
    t0 : int, optional
        Start sampling the data after t0 timesteps, this should be 2 days 
        after model start to ignore model spin up. Defaults to 4.

    Returns
    ----------
    freq : ndarray
        Array of sample frequencies
    w_psd_xy : ndarray
        Power spectral density of w_latlon combining the x and y directions into one array
    """
    # Initalize dimentions parameters
    nt = w_latlon.shape[0] - t0
    nlon = w_latlon.shape[1]
    nlat = w_latlon.shape[2]
    nkmax = np.max([nlon, nlat])
    nkmin = np.min([nlon, nlat])
    # initialize power spectrum array for each timestep
    # assuming nlon==nlat and power spectrum return an
    # array with size 19 (may change based on your setup)
    w_t = w_latlon.isel(time=t0).values
    freq_x, _ = welch(w_t[0,:], fs=fs, detrend='linear', nperseg=nkmin, **kwargs)
    nfreqx = len(freq_x)
    freq_y, _ = welch(w_t[:,0], fs=fs, detrend='linear', nperseg=nkmin, **kwargs)
    nfreqy = len(freq_y)
    w_psd_x = np.zeros((nt,nlon,nfreqx))
    w_psd_y = np.zeros((nt,nlat,nfreqy))
    # start at t0=4 (day 2) to allow for spin up
    for t in range(t0,t0+nt):
        tt = t-t0 # for indexing power spectrum array
        w_t = w_latlon.isel(time=t).values
        # Estimate power spectral density using Welch’s method across the x and y direction
        # combine x and y information into a single array for calculating the w psd
        for k in range(nkmax):
            if k<(nlon-1):
                freq_x, w_psd_x[tt,k,:] = welch(w_t[k,:], fs=fs, detrend='linear', nperseg=nkmin, **kwargs)
            if k<(nlat-1):
                freq_y, w_psd_y[tt,k,:] = welch(w_t[:,k], fs=fs, detrend='linear', nperseg=nkmin, **kwargs)
    return (freq_x, freq_y, w_psd_x, w_psd_y)

def calc_geodesic_distance(latlon0, latlon1):
    """ Calculates the distance between two points
        given in (lon) coords, returns the distance
        in kilometers
        Uses metpy package
    """
    return( geodesic(latlon0, latlon1).km * units.km)

def plot_model_w_psd(freq, psd, ax=None, color='b', label="SCREAM", **kwargs):
    """
    Plots the statistics of the power spectrum density from model 
    output on a lat-lon grid. The PSD is plotted as median (solid line),
    shading between 25th and 75th percentiles, dashed lines denote the
    5th and 95th percentiles.
    
    See regrid_w() and calc_pwd_welch() whose outputs are in the correct
    format for input into this function.

    Parameters
    ----------
    freq : array-like 
        Array of sample frequencies
    psd : array-like
        Power spectral density (m2/s3) of vertical velocity (m/s)
    ax : matplotlib.pyplot.axes, Defaults to None
        Axes provided for plotting, if None then an axes will be generated
        and returned as output
    color : string, optional 
        Defines the color of the lines/shading for the model PSD
    **kwargs : optional kwargs
        passed into matplotlib.pyplot.axes object for plotting

    Returns
    --------
    ax : matplotlib.pyplot.axes
        Returns the axes on which the data was plotted
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4), constrained_layout=True)
    # Take statistics over all times, latitudes, and longitudes
    ax.loglog(1/(freq), np.median(psd,axis=(0,1)), color, label=label, *kwargs)
    # ax.loglog(1/(freq), np.quantile(psd,0.05,axis=(0,1)), color, linestyle='dashed', **kwargs)
    # ax.loglog(1/(freq), np.quantile(psd,0.95,axis=(0,1)), color, linestyle='dashed', **kwargs)
    ax.fill_between(1/(freq),
                    np.quantile(psd,0.25,axis=(0,1)),
                    np.quantile(psd,0.75,axis=(0,1)),
                    color=color, alpha=.25, **kwargs)
    ax.set(xlabel='Wavelength (km)',
           ylabel="Power (m$^2$ s$^{-3}$)",
           xlim=(0.5,100), ylim=(1e-7,10),
        #    xticks=np.arange(1,101,10),
        #    xscale='log', 
           yscale='log',
           )
    return ax

def plot_attrex_w_pwd(file_name=None, ax=None, **kwargs):
    """
    Plots the statistics of the power spectrum density from ATTREX data 
    output as a function of the wavelength. The PSD is plotted as median 
    (solid line), shading between 25th and 75th percentiles, dashed 
    lines denote the 5th and 95th percentiles.
    
    See regrid_w() and calc_pwd_welch() whose outputs are in the correct
    format for input into this function.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes, Defaults to None
        Axes provided for plotting, if None then an axes will be generated
        and returned as output
    **kwargs : optional kwargs
        passed into matplotlib.pyplot.axes object for plotting

    Returns
    --------
    ax : matplotlib.pyplot.axes
        Returns the axes on which the data was plotted
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))
    attrex_ds = open_attrex_twp(file_name)
    #Plot observations that were loaded from a python library earlier
    ax.loglog(attrex_ds.wavelength,attrex_ds.p50,'k', label="ATTREX")
    # ax.loglog(attrex_ds.wavelength,attrex_ds.p05,'k--')
    # ax.loglog(attrex_ds.wavelength,attrex_ds.p95,'k--')
    ax.fill_between(attrex_ds.wavelength,attrex_ds.p25,attrex_ds.p75,color='k',alpha=.25)
    ax.set(xlabel='Wavelength (km)',
           ylabel=r"Power (m$^2$ s$^{-3}$)",
           xlim=(3,100), ylim=(1e-7,10),
           xticks=np.arange(10,101,10),
        #    xscale='log',
           yscale='log',
           )
    return ax


######################################
#### ---    open from file    --- ####
######################################

def open_attrex_twp(file_dir=None):
    """Loads ATTREX spectrum detrended data from TWP region,
       Requires file generated by Rachel Atlas: 
           "ATTREX_spectrum_detrended_100km_TWP_only"

       Returns:
       xarray.Dataset  : an xarray dataset with dimentions for x
                         then the percentiles are saved as different variables
                         'p05', 'p25', 'p50', 'p75', 'p95'
    """
                         
    #This loads saved ATTREX spectra
    if file_dir is None:
        fn='ATTREX_spectrum_detrended_100km_TWP_only'
    else:
        fn=file_dir+'/ATTREX_spectrum_detrended_100km_TWP_only'
    with open(fn,'rb') as fid:
      Lib=pickle.load(fid)
    
    #load ATTREX spectra statistics
    xaxis=Lib['xaxis'][:]
    p05=xr.DataArray(Lib['p05'][:], 
                     dims=['wavelength'],
                     coords={'wavelength':xaxis},
                     attrs={'long_name':'5th percentile of ATTREX power spectrum'})
    p25=xr.DataArray(Lib['p25'][:],
                     dims=['wavelength'],
                     coords={'wavelength':xaxis},
                     attrs={'long_name':'25th percentile of ATTREX power spectrum'})
    p50=xr.DataArray(Lib['median'][:],
                     dims=['wavelength'],
                     coords={'wavelength':xaxis},
                     attrs={'long_name':'50th percentile of ATTREX power spectrum'})
    p75=xr.DataArray(Lib['p75'][:], 
                     dims=['wavelength'],
                     coords={'wavelength':xaxis},
                     attrs={'long_name':'75th percentile of ATTREX power spectrum'})
    p95=xr.DataArray(Lib['p95'][:], 
                     dims=['wavelength'],
                     coords={'wavelength':xaxis},
                     attrs={'long_name':'95th percentile of ATTREX power spectrum'})

    ds = xr.Dataset({'p05':p05,'p25':p25,
                     'p50':p50, 
                     'p75':p75, 'p95':p95},
                   attrs={'long_name':'ATTREX_spectrum_detrended_100km_TWP_only',
                          'history':'generated by Rachel Atlas using ATTREX campaign data',
                          'region':'Tropical Western Pacific (TWP)'})
    return ds

def open_w_dataarray(file_name, omega_var="omega", tair_var="T_mid"):
    """Returns xarray.DataArray with vertical velocity data from
       given a dataset with omega, temperature, pressure
       
       Set up for SCREAM formatted files (i.e., all variables and 
       timesteps in a single file)
    """
    # load only the variables we need
    ds = xr.open_dataset(file_name)[[omega_var,tair_var,'lat','lon','time']]
    if 'lev' in ds[omega_var].dims:
        w_ds = calc.omega2w(ds[omega_var].metpy.quantify(), ds.lev * units.hPa, ds[tair_var].metpy.quantify())
    else:
        lev = int(omega_var.split("_")[-1].split("hPa")[0])
        print(lev * units.hPa)
        w_ds = calc.omega2w(ds[omega_var].metpy.quantify(), lev * units.hPa, ds[tair_var].metpy.quantify())
    ds['W'] = w_ds
    if len(ds.lon.shape)>1:
        ds['lon'] = fix_longitudes(ds['lon'].isel(time=0))
        ds['lat'] = ds['lat'].isel(time=0)
    else:
        ds['lon'] = fix_longitudes(ds['lon'])
        ds['lat'] = ds['lat']
    ds = ds.set_coords(['lat','lon'])
    return(ds)

def fix_longitudes(lon):
    """fix longitude so that it is always from -180 to 180 
       (not 0 to 360)"""
    return ((lon + 180) % 360) - 180

