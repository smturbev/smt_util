
#!/ascldap/users/smturbe/.conda/envs/smt_plt/bin/python

"""
plot_functions.py

collection of useful plots (maps and stats analysis) for analysis
of scream output.

Some pieces of this code was taken from brhillman/esmplot
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from scipy.interpolate import griddata
import xarray as xr
import xoak



def setup_map_carrm(ncols, nrows, nax, fig, proj=ccrs.PlateCarree(), tight=False):
    ax = fig.add_subplot(ncols, nrows, nax, projection=proj)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.right_labels=False
    gl.top_labels=False
    if tight:
        ax.set_extent([-120, -114, 35, 39])
    else:
        ax.set_extent([-125, -112, 32, 42])
    return ax

def setup_map_global(fig, proj=ccrs.PlateCarree()):
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
    ax.gridlines(draw_labels=True)
    return ax

def fix_longitudes(lon):
    """fix longitude so that it is always from -180 to 180 
       (not 0 to 360)"""
    return ((lon + 180) % 360) - 180

def plot_map(x, y, data, ax=None, proj=None, method='tri', nlon=360, nlat=180, clabel=None, **kwargs):
    """plot a geologic map using triangulation"""
    if ax is None:
        ax = plt.gca()
    if proj is None:
        proj = ccrs.PlateCarree()
    
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False

    x = fix_longitudes(x)

    if method in ['tri', 'triangulation']:
        im = ax.tripcolor(x, y, data, transform=proj, **kwargs)
    # elif method=='regrid':
    #     xi = np.linspace(-180, 180, int(nlon))
    #     yi = np.linspace(-90, 90, int(nlat))
    #     data_regridded = griddata((x, y), data, (xi[None,:], yi[:,None]), method='nearest')
    #     pl = ax.pcolormesh(xi, yi, data_regridded, transform=ccrs.PlateCarree(), **kwargs)
    else:
        raise ValueError(f'method {method} not known')
    
    # Add a colorbar
    cb = plt.colorbar(im, ax=ax, label=clabel)

    # Return plot and colorbar
    return ax, cb

def xoak_kdtree_sel(ds, lats, lons):
    """
    Uses xoak to select data with irregular dimensions

    Parameters
    ----------
    ds : `xarray.Dataset`
      Dataset with your data, with lat/lon dims in one-dimension

    lats : list
      List of latitudes to select

    lons : list
      List of longitudes to select

    Returns
    -------
    ds : `xr.Dataset`
      Dataset subset based on input lats/lons
    """

    # Checks to see if this has an xoak index - if not, create one using the extension
    if not ds.xoak.index:
        ds.xoak.set_index(("lat", "lon"), "scipy_kdtree")

    # Return the selected datasets
    return ds.xoak.sel(lat=xr.Variable("ncol", lats), lon=xr.Variable("ncol", lons))



def dennisplot(stat, olr, alb, var=None, xbins=None, ybins=None, units=None, savename=None,
               levels=None, cmap=cm.ocean_r, ax=None, save=False, colorbar_on=True, fs=20, contour_on=False,
               ):
    ''' Returns axis with contourf of olr and albedo.
    
    Parameters:
        - stat (str)   : - 'difference' returns contourf of the difference between the first minus the second in the tuple
                         - 'density' returns density plot of olr-alb joint histogram (pdf), or
                         - statistic for scipy.stats.binned_statistic_2d
        - olr (array)  : 1-D array of OLR values (from 85-310 W/m2), 
        - alb (array)  : 1-D array of Albedo values (from 0-1),
        - var (array)  : 1-D array (var is optional if stat=density or difference)
        - colorbar_on (bool)
                       : returns a tuple of ax, mappable_countour if False
        - xbins (list,int,array) : provide bins for albedo
        - ybins (list,int,array) : provide bins for OLR
        - units (str)   : for colorbar label, if var is not None
        - ax (matplotlib.axis)   : provide axis for plotting
        - contour_on (bool)      : returns the outlines/contour of the joint histogram if True
        - levels (array)         : levels for contour/contourf
                       
    Returns: 
        - ax (plt.axis): axis with plot 
        - cs (mappable): returned value from plt.contourf, if colorbar_on = False
        
    Note: Values for mean sw downward flux at toa from 
              http://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/Lectures/Lecture11%20--%20Insolation.html. 
    '''
    if xbins is None:
        xbins = np.linspace(70,320,26)
    if ybins is None:
        ybins = np.linspace(0,0.8,33)
    if levels is None:
        if stat=="difference":
            levels = np.arange(-0.9,1,0.2)
        else:
            levels = np.arange(-3,-1.2,0.1)
    if stat=="difference":
        print("difference")
        olr0, olr1 = olr
        alb0, alb1 = alb
        olr0 = olr0[~np.isnan(alb0)]
        alb0 = alb0[~np.isnan(alb0)]
        alb0 = alb0[~np.isnan(olr0)]
        olr0 = olr0[~np.isnan(olr0)]
        olr1 = olr1[~np.isnan(alb1)]
        alb1 = alb1[~np.isnan(alb1)]
        alb1 = alb1[~np.isnan(olr1)]
        olr1 = olr1[~np.isnan(olr1)]
        hist0, xedges, yedges = np.histogram2d(olr0,alb0,bins=(xbins,ybins))
        nan_len = np.sum(~np.isnan(alb0))
        hist0 = hist0/nan_len
        print(nan_len)
        hist1, xedges, yedges = np.histogram2d(olr1,alb1,bins=(xbins,ybins))
        nan_len = np.sum(~np.isnan(alb1))
        hist1 = hist1/nan_len
        print(nan_len)
        binned_stat = hist0-hist1
    else:
        if (olr.shape!=alb.shape) and (var is not None):
            raise Exception("shapes don't match: olr %s, alb %s, %s %s."%(olr.shape, alb.shape, var_name, var.shape))
        elif var is not None:
            if (olr.shape!=var.shape) or (alb.shape!=var.shape):
                raise Exception("shapes don't match: olr %s, alb %s, %s %s."%(olr.shape, alb.shape, var_name, var.shape))
        elif (olr.shape!=alb.shape):
            raise Exception("shapes of alb and olr don't match: %s != %s"%(alb.shape, olr.shape))
        olr = olr[~np.isnan(alb)]
        if stat!='density':
            var = var[~np.isnan(alb)]
        alb = alb[~np.isnan(alb)]
        alb = alb[~np.isnan(olr)]
        if stat!='density':
            var = var[~np.isnan(olr)]
        olr = olr[~np.isnan(olr)]
        if stat!='density':
            alb = alb[~np.isnan(var)]
            olr = olr[~np.isnan(var)]
            var = var[~np.isnan(var)]
        if stat=='density':
            # check for nans
            binned_stat, xedges, yedges = np.histogram2d(olr,alb,bins=(xbins,ybins))
            nan_len = xr.DataArray(alb).count().values
            binned_stat = binned_stat/(nan_len)
            print(nan_len)
        else: 
            var = var[~np.isnan(olr)]
            binned_stat, xedges, yedges, nbins = stats.binned_statistic_2d(olr, alb, var, 
                                                                          bins=(xbins,ybins), statistic=stat)
    xmids, ymids = (xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2
    if ax is None:
        ax = plt.gca()
    if stat=="difference":
        csn = ax.contourf(xmids, ymids, binned_stat.T*100, levels, cmap=cmap, extend='both')
        if contour_on:
            con_levs = np.arange(-3,-1.2,0.1)
            con = ax.contour(xmids, ymids, np.log10(hist0.T), con_levs, 
                             colors='k', linestyles='solid', linewidths=1)
    elif stat=="density":
        csn = ax.contourf(xmids, ymids, np.log10(binned_stat.T), levels, cmap=cmap, extend='both')
        co = ax.contour(csn, colors='k', linestyles='solid', linewidths=1)
    else:
        csn = ax.contourf(xmids, ymids, (binned_stat.T), levels, cmap=cmap, extend='both')
        co = ax.contour(csn, colors='k', linestyles='solid', linewidths=1)
    ax.grid()
    ax.set_xticks([100,150,200,250,300])
    ax.set_ylim([0.05,0.8])
    ax.set_xlim([80,310])
    ax.set_xlabel('OLR(W m$^{-2}$)', size=fs)
    ax.set_ylabel('Albedo', size=fs)
    ax.tick_params(axis='both',labelsize=fs)

    # plot the colorbar
    if colorbar_on:
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        cb = plt.colorbar(csn, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=fs-4)
        if stat=="density":
            cb.set_label('log10(pdf)', fontsize=fs)
        elif stat=="difference":
            cb.set_label('pdf % difference', fontsize=fs)
            cb.set_ticks((levels[1:]+levels[:-1])/2)
        else:
            cb.set_label('{} ({})'.format(stat, units), fontsize=fs)
    if save:
        plt.savefig(savename, bbox_inches="tight")
        print(f'    saved as {savename}')
    if colorbar_on:
        ret = ax
    elif contour_on:
        ret = ax, csn, con
    else:
        ret = ax, csn
    return ret