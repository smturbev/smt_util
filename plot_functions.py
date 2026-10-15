
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
from geopy.distance import geodesic
from scipy.interpolate import griddata
from metpy import units as mpunits
from metpy.interpolate import cross_section as mp_cross_section

import xarray as xr
import pandas as pd
import matplotlib.colors as mcolors


# Create custom colormap for ice water path (transparent at low values)
cdict = {
    'red':   [(0, 1, 1), (1, 1, 1)],
    'green': [(0, 1, 1), (1, 1, 1)],
    'blue':  [(0, 1, 1), (1, 1, 1)],
    'alpha': [(0, 0, 0), (1, 1, 1)],  # key: alpha 0 → 1
}
my_cm = mcolors.LinearSegmentedColormap(
    'my_cm', cdict
)

def fix_longitudes(lon):
    """fix longitude so that it is always from -180 to 180 
       (not 0 to 360)"""
    return ((lon + 180) % 360) - 180

def get_latlon_box(ds, lat0, lat1, lon0, lon1, lats=None, lons=None):
    """Returns a subsection of the dataset within the lat-lon box"""
    if lats is None:
        # Get lats and lons
        lats = ds.lat.values
    if lons is None:
        lons = fix_longitudes(ds.lon.values)

    # Normalize bounds
    lat_min, lat_max = min(lat0, lat1), max(lat0, lat1)
    lon_min, lon_max = min(lon0, lon1), max(lon0, lon1)

    # Latitude mask
    lat_mask = (lats >= lat_min) & (lats <= lat_max)

    # Longitude mask with wrap handling
    if lon0 <= lon1:
        lon_mask = (lons >= lon0) & (lons <= lon1)
    else:
        # box crosses dateline in normalized coordinates
        lon_mask = (lons >= lon0) | (lons <= lon1)

    col_mask = lat_mask & lon_mask
    sel_idxs = np.where(col_mask)[0]

    return ds.isel({"ncol":sel_idxs})

def get_cross_section(da, start_latlon, end_latlon, n_points=100, hres=None):
    """
    Extracts a cross-section from an unstructured (ncol, lev) DataArray.
    
    Parameters:
    -----------
    da : xarray.DataArray
        The input data with dimensions (ncol, lev).
    start_latlon : tuple
        (lat0, lon0)
    end_latlon : tuple
        (lat1, lon1)
    n_points : int (optional, must have either n_points or hres)
        Number of sampling points along the line.
    hres : float (optional, must have either n_points or hres)
        Resolution in km for the grid spacing along the trajectory of the cross section.
        If hres is not None, it overwrites the value of n_points.

    --------
    Usage example:
        cs = get_cross_section(ds.T_mid, (24.0, -118.0), (43.0, -118.5), n_points=200)
        cs.plot(x='distance', y='lev', yincrease=False)
    """
    dist_km = geodesic(start_latlon, end_latlon).km
    # Get the geodesic distance between start and end points in km
    if hres is not None:
        n_points = int(dist_km//hres)
        print(f"Distance between {start_latlon} and {end_latlon} is {dist_km} km ({n_points} points with hres={hres} km).")

    # 1. Generate points along the cross-section line
    lats_line = np.linspace(start_latlon[0], end_latlon[0], n_points)
    lons_line = np.linspace(start_latlon[1], end_latlon[1], n_points)
    
    # 2. Find nearest ncol for each point (Nearest Neighbor)
    # Using broadcasting to find the minimum distance for every point on the line
    # Note: For very large datasets, consider a KDTree for performance.
    lat_diff = da.lat.values[None, :] - lats_line[:, None]
    lon_diff = da.lon.values[None, :] - lons_line[:, None]
    dist_sq = lat_diff**2 + lon_diff**2
    
    nearest_ncol_indices = dist_sq.argmin(axis=1)
    
    # 3. Subset the DataArray
    # This maps each 'distance' step to the best 'ncol'
    section = da.isel(ncol=nearest_ncol_indices)
    uniq_ncols = len(np.unique(nearest_ncol_indices))
    if (uniq_ncols/len(nearest_ncol_indices)) < 0.7:
        print('----WARNING----')
        print('Number of unique points:', uniq_ncols)
        print('Fraction of unique points',uniq_ncols/len(nearest_ncol_indices))
        raise Exception('Not getting uique points for the cross section, see above print statemtn in output')
    
    # 4. Finalize dimensions and coordinates
    # Rename ncol to 'distance' to distinguish from the original grid
    section = section.rename({'ncol': 'index'})
    
    # Add coordinate metadata for plotting
    section = section.assign_coords(
        index=np.arange(n_points),
        lat_section=('index', lats_line),
        lon_section=('index', lons_line)
    )

    section = section.assign_coords(
        distance=('index',np.linspace(0,dist_km,n_points))
    )
    
    return section

def get_cross_section_latlon(da, start_latlon, end_latlon, n_points=100, hres=None):
    """
    Extracts a cross-section from a regular lat-lon DataArray.
    
    Parameters:
    -----------
    da : xarray.DataArray
        The input data with dimensions (lat, lon, lev).
    start_latlon : tuple
        (lat0, lon0)
    end_latlon : tuple
        (lat1, lon1)
    n_points : int (optional, must have either n_points or hres)
        Number of sampling points along the line.
    hres : float (optional, must have either n_points or hres)
        Resolution in km for the grid spacing along the trajectory of the cross section.
        If hres is not None, it overwrites the value of n_points.

    --------
    Usage example:
        cs = get_cross_section(ds.T_mid, (24.0, -118.0), (43.0, -118.5), n_points=200)
        cs.plot(x='distance', y='lev', yincrease=False)
    """
    dist_km = geodesic(start_latlon, end_latlon).km
    # Get the geodesic distance between start and end points in km
    if hres is not None:
        n_points = int(dist_km//hres)
        print(f"Distance between {start_latlon} and {end_latlon} is {dist_km} km ({n_points} points with hres={hres} km).")

    da = da.metpy.parse_cf()
    print("da with metpy parse cf", da)
    section = mp_cross_section(da, start_latlon, end_latlon, steps=n_points, interp_type="nearest")
    
    section = section.assign_coords(
        distance=('index',np.linspace(0,dist_km,n_points))
    )

    return section

def set_secondary_axis_latlon(ax, start, end, x0=0, x1=1):
    ax2 = ax.secondary_xaxis(location=-0.1)
    ax2.set_xticks([x0,x1])
    ax2.set_xticklabels([f'{start[0]}$^\circ$N, {start[1]}$^\circ$E',f'{end[0]}$^\circ$N, {end[1]}$^\circ$E'])
    ax2.spines["bottom"].set_visible(False) # Hide spine line
    return ax2

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
    # ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
    ax.gridlines(draw_labels=True)
    return ax

def setup_map_global(fig, nrow, ncol, idx, region_extent=[-180, 180, -90, 90], proj=ccrs.PlateCarree()):
    ax = fig.add_subplot(nrow, ncol, idx, projection=proj)
    ax.set_global()
    ax.coastlines(resolution='50m', linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.25)
    ax.gridlines(draw_labels=True)
    ax.set_extent(region_extent, crs=proj)
    return ax

def plot_map(x, y, data, ax=None, proj=None, method='tri', nlon=360, nlat=180, clabel=None, **kwargs):
    """plot a geologic map using triangulation"""
    if proj is None:
        proj = ccrs.PlateCarree()
    if ax is None:
        fig = plt.figure()
        ax = setup_map_global(fig, proj=proj)

    x = fix_longitudes(x)

    if method in ['tri', 'triangulation']:
        im = ax.tripcolor(x, y, data, transform=proj, **kwargs)
    elif method=='regrid':
        xi = np.linspace(-180, 180, int(nlon))
        yi = np.linspace(-90, 90, int(nlat))
        data_regridded = griddata((x, y), data, (xi[None,:], yi[:,None]), method='nearest')
        im = ax.pcolormesh(xi, yi, data_regridded, transform=ccrs.PlateCarree(), **kwargs)
    else:
        raise ValueError(f'method {method} not known')
    
    # Add a colorbar
    cb = plt.colorbar(im, ax=ax, label=clabel)

    # Return plot and colorbar
    return ax, cb

def undo_rolling_average_era(da,time_ind="valid_time"):
    """radiation variables are given as accumulated"""
    # new time index will be half way between old time indices
    ds_time0 = pd.to_datetime(da[time_ind].values)
    dt = ds_time0[1]-ds_time0[0]
    print(dt.total_seconds(), 'seconds')
    ds_time05 = ds_time0[:-1] + dt/2
    print(ds_time0)
    print(ds_time05)
    # undo accumulation (accumulation over output timestep) 
    # divide by the seconds in timestep to get mean value in each hour
    # down is positive in reanalysis, so change to up is pos coords
    unacc_array = -1*(da.values)/3600  #dt.total_seconds()
    da_new = xr.DataArray(unacc_array, dims=[time_ind,'latitude','longitude'],
                            coords={time_ind:ds_time0,
                                    'latitude':da.latitude,
                                    'longitude':da.longitude},
                            attrs={'long_name':f'Unaccumulated {da.attrs['long_name']}',
                                    'units':'W m**-2'})
    return da_new


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

def open_uv_from_rotated_um_datset(um_file_name):
    """Code from Sid Gumber (BAS) to convert from polar 
    lat lon coordinates to regular lat lon"""
    ds3 = xr.open_dataset(
        um_file_name,
        chunks="auto" # lazily loads the file 
    )
    u = ds3["U"]
    v = ds3["V"]

    # Get rotated-pole information
    grid_mapping_var = next(
        (
            name for name in ["rotated_pole", "rotated_latitude_longitude"]
            if name in ds3
        ),
        None
    )

    if grid_mapping_var is None:
        raise ValueError("Could not find rotated-pole grid mapping in ds3")

    pole_lat = ds3[grid_mapping_var].attrs["grid_north_pole_latitude"]
    pole_lon = ds3[grid_mapping_var].attrs["grid_north_pole_longitude"]

    print("Rotated pole latitude :", pole_lat)
    print("Rotated pole longitude:", pole_lon)

    gm = ds3["rotated_latitude_longitude"]

    rotated_cs = iris.coord_systems.RotatedGeogCS(
        grid_north_pole_latitude=float(
            gm.attrs["grid_north_pole_latitude"]
        ),
        grid_north_pole_longitude=float(
            gm.attrs["grid_north_pole_longitude"]
        )
    )

    print(rotated_cs)

    u_cube = u.to_iris()
    v_cube = v.to_iris()

    # Attach the rotated-pole coordinate system
    for cube in (u_cube, v_cube):
        cube.coord("grid_longitude").coord_system = rotated_cs
        cube.coord("grid_latitude").coord_system = rotated_cs

    print("U x CS:", u_cube.coord(axis="x").coord_system)
    print("U y CS:", u_cube.coord(axis="y").coord_system)

    # Rotate
    u_geo_cube, v_geo_cube = cart.rotate_winds(
        u_cube,
        v_cube,
        target_cs=iris.coord_systems.GeogCS(6371229.0)
    )

    print("Rotation graph/result created.")

    u_geo_cube.data = u_geo_cube.core_data()
    v_geo_cube.data = v_geo_cube.core_data()

    print("Computation complete.")

    # Back to Xarray
    u_geo = xr.DataArray.from_iris(u_geo_cube)
    v_geo = xr.DataArray.from_iris(v_geo_cube)
    #pmomx_geo = xr.DataArray.from_iris(pmomx_geo_cube)#
    #pmomy_geo = xr.DataArray.from_iris(pmomy_geo_cube)

    print("u_geo shape:", u_geo.shape)
    print("u_geo dims:", u_geo.dims)

    print("v_geo shape:", v_geo.shape)
    print("v_geo dims:", v_geo.dims)
    return u_geo, v_geo