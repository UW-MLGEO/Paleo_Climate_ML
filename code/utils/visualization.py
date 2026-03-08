"""
Visualization utilities for climate data.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import rioxarray as rxr

from typing import Optional, Tuple, Any

"""
For CNN plotting.
"""
import numpy as np
import matplotlib as mpl
import cartopy.feature as cfeature
from shapely.geometry.polygon import LinearRing

import cartopy as ct
import warnings
import sklearn
from sklearn import metrics


def setup_map_axes(
    figsize: Tuple[float, float] = (14, 8),
    projection: Optional[Any] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure and axes with cartopy projection for map plotting.
    
    Parameters
    ----------
    figsize : tuple of float, default (14, 8)
        Figure size (width, height) in inches
    projection : cartopy projection, optional
        Map projection. Defaults to PlateCarree
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes or GeoAxes
        Axes object with projection
    """
    if projection is None:
        projection = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.coastlines(linewidth=1)
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    return fig, ax


def plot_spatial_map(
    data: xr.DataArray,
    title: str = "",
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = True,
    figsize: Tuple[float, float] = (14, 8),
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a spatial map of climate data.
    
    Parameters
    ----------
    data : xr.DataArray
        2D data to plot (should have lat/lon dimensions)
    title : str, optional
        Plot title
    cmap : str, default "coolwarm"
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    robust : bool, default True
        Use robust color limits (2nd and 98th percentiles)
    figsize : tuple of float, default (14, 8)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = setup_map_axes(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Reduce time and member dimensions if present
    plot_data = data.copy()
    if "time" in plot_data.dims:
        plot_data = plot_data.mean("time")
    if "member" in plot_data.dims:
        plot_data = plot_data.isel(member=0)
    
    # Plot
    plot_kwargs = {
        'ax': ax,
        'transform': ccrs.PlateCarree(),
        'cmap': cmap,
        'robust': robust,
        'x': 'lon',
        'y': 'lat'
    }
    
    if vmin is not None:
        plot_kwargs['vmin'] = vmin
    if vmax is not None:
        plot_kwargs['vmax'] = vmax
    
    plot_data.plot(**plot_kwargs)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    return fig, ax


def plot_zonal_mean(
    data: xr.DataArray,
    title: str = "",
    cmap: str = "coolwarm",
    figsize: Tuple[float, float] = (14, 5),
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot zonal mean (latitude vs time) of climate data.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with time and latitude dimensions
    title : str, optional
        Plot title
    cmap : str, default "coolwarm"
        Colormap name
    figsize : tuple of float, default (14, 5)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Compute zonal mean
    zonal_data = data.copy()
    if "lon" in zonal_data.dims:
        zonal_data = zonal_data.mean("lon")
    elif "longitude" in zonal_data.dims:
        zonal_data = zonal_data.mean("longitude")
    
    # Reduce member dimension if present
    if "member" in zonal_data.dims:
        zonal_data = zonal_data.mean("member")
    
    # Get time as years
    if "time" in zonal_data.coords:
        years = zonal_data["time"].dt.year + (zonal_data["time"].dt.month - 1) / 12
        lat = zonal_data["lat"].values if "lat" in zonal_data.coords else zonal_data["latitude"].values
        
        im = ax.pcolormesh(years, lat, zonal_data.T, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label=zonal_data.attrs.get('units', ''))
        
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    return fig, ax

"""
Plotting utilities adapted from the CNN repository
"""

# Settings:

# Matplotlib colors
npcols = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
]

warnings.filterwarnings("ignore")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

data_crs = ccrs.PlateCarree()
proj_Global = ccrs.EqualEarth(central_longitude=200)

"""
Functions
"""


def plot_metrics(history, metric):
    """
    history options: loss, val_loss, mae, val_mae
    
    Shows how the model improves during training.
    
    1. plot_metrics(history, 'loss')
    Shows mean squared error values for training loss and validation loss.
    Shows how close predicted values are to the actual values.

    2. plot_metrics(history, "mae")
    Mean absolute error in W/m^2.

    imin = epoch with lowest validation loss
    """
    imin = np.argmin(history.history['val_loss'])

    plt.plot(history.history[metric], label='training')
    plt.plot(history.history['val_' + metric], label='validation')
    plt.title(metric)
    plt.axvline(x=imin, linewidth=.5, color='gray', alpha=.5)
    plt.legend()

def plot_metrics_panels(history, predictions):
    # BASELINES TO BEAT
    baseline_mae = np.mean(np.abs(predictions["labels_test"] - 0.0))
    baseline_mse = np.mean((predictions["labels_test"] - 0.0)**2)

    # DO THE PLOTTING
    plt.subplots(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plot_metrics(history, 'loss')
    plt.ylim(0, .5)
    plt.axhline(y=baseline_mse, color="gray", linestyle="--", label="baseline")

    plt.subplot(1, 4, 2)
    plot_metrics(history, "mae")
    plt.ylim(0, .5)
    plt.axhline(y=baseline_mae, color="gray", linestyle="--", label="baseline")

    plt.legend()


def plot_pred_vs_truth(predictions, settings, ms=5, label=''):

    """
    Shows true CRE (x-axis) vs. predicted CRE (y-axis).
    A diagonal line represents be a perfect prediction.
    """

    mse = sklearn.metrics.mean_squared_error(predictions["labels_val"], predictions["pred_val"])
    if np.isnan(mse):
        mse = 0.
    plt.plot(predictions["labels_val"], predictions["pred_val"], '.', alpha=.5,
                label=label + ' (MSE=' + str(mse.round(3)) + ')', markersize=ms)

    plt.plot((-10, 10), (-10, 10), '--k', alpha=.5)
    plt.ylim(-8, 3)
    plt.xlim(-8, 3)
    plt.ylabel('predicted')
    plt.xlabel('truth')
    plt.legend(frameon=False, fontsize=8)
    plt.gca().set_aspect('equal')

def setup_figure(nCols=1,nRows=1,size=(15,15),mask=True):

    """
    This function makes sure that all plots we make have the same projection and layout.
    
    Example usage to make two maps:
    
    fig, ax = setup_figure(nCols=2, size=(12,5))

    ax[0].contourf(...)
    ax[1].contourf(...)
    """

    map_proj = proj_Global

    land_feature = cfeature.NaturalEarthFeature(

        "If mask = True, then the land will appear grey",

        category='physical',
        name='land',
        scale='50m',
        facecolor='gray',
        edgecolor='k',
        linewidth=.25,
    )

    fig = plt.figure(figsize=size)
    if nCols==1 and nRows==1:

        """
        +----------------+
        |                |
        |   Global map   |
        |                |
        +----------------+
        """
        ax = fig.add_subplot(1, 1, 1, projection=map_proj)

        ax.coastlines('50m', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )

        if mask:
            ax.add_feature(land_feature)
    elif nCols > 1 and nRows > 1:
        
        """
        +---------+---------+
        | map 1   | map 2   |
        +---------+---------+
        | map 3   | map 4   |
        +---------+---------+
        """
        ax = np.empty((nCols,nRows),dtype=object)
        iF = 1
        for jj in range(nRows):
            for ii in range(nCols):
                ax[ii,jj] = fig.add_subplot(nRows, nCols, iF, projection=map_proj)

                ax[ii,jj].coastlines('50m', linewidth=0.8)
                ax[ii,jj].tick_params(axis='both', which='major', labelsize=10)
                ax[ii,jj].gridlines(
                    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
                )
                if mask:
                    ax[ii,jj].add_feature(land_feature)
                iF += 1
    elif nCols > 1:
        """
        +-------+-------+-------+
        | map1  | map2  | map3  |
        +-------+-------+-------+
        """
        ax = np.empty((nCols),dtype=object)
        for ii in range(nCols):
            ax[ii] = fig.add_subplot(1, nCols, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if mask:
                ax[ii].add_feature(land_feature)
    elif nRows > 1:
        """
        +-------+
        | map1  |
        +-------+
        | map2  |
        +-------+
        | map3  |
        +-------+
        """
        ax = np.empty((nRows),dtype=object)
        for ii in range(nRows):
            ax[ii] = fig.add_subplot(nRows, 1, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if mask:
                ax[ii].add_feature(land_feature)

            
    return fig, ax

def add_mask(ax,region,lon=None,lat=None):
    
    """
    Adds a mask for all land, a predefined region, or from a mask
    stored in a NetCDF file
    """

    if region == "land":
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='50m',
            facecolor='gray',
            edgecolor='k',
            linewidth=.25,
        )
        ax.add_feature(land_feature)
    elif region[:4] == "reg_":

        regions_range_dict = regions.get_region_dict(region)

        min_lon, max_lon = regions_range_dict["lon_range"]
        min_lat, max_lat = regions_range_dict["lat_range"]
        region = [min_lon, min_lat, max_lon, max_lat]

        add_square(ax,region,data_crs,facecolor='gray')

    elif region[-3:] == ".nc":

        mask = xr.load_dataarray(SHAPE_DIRECTORY + region).to_numpy()

        mask[mask>0.5] = np.nan
        mask_cyc, lons_cyc = add_cyclic_point(mask, coord=lon)
        ax.pcolormesh(lons_cyc,lat,mask_cyc,cmap="gray")

    return ax

def add_square(ax,region,crs,**kwargs):
    # Region = lon1, lat1; lon2, lat2

    """
    Example:
    +----------------------+
    |                      |
    |      ┌─────────┐     |
    |      │ region  │     |
    |      └─────────┘     |
    |                      |
    +----------------------+
    """

    lons = [region[0],region[2],region[2],region[0]]
    lats = [region[1],region[1],region[3],region[3]]
    ring = LinearRing(list(zip(lons,lats)))

    ax.add_geometries([ring],crs,**kwargs)

    return ax


def add_loc_square(ax,settings,**kwargs):
    """
    Use this to add a rectangle to a region without
    having to put in specific coordinates
    """
    if "mask_region" in settings.keys():
        maskout=settings["mask_region"]
    else:
        return ax

    if maskout == "indonesia":
        min_lon, max_lon = 75., 130.
        min_lat, max_lat = -15., 10.
    elif maskout == "westpacific":
        min_lon, max_lon = 170., 230.
        min_lat, max_lat = -30., -5.
    elif maskout == "eastpacific":
        min_lon, max_lon = 210., 280.
        min_lat, max_lat = -10., 10.
    elif maskout == "caribbean":
        min_lon, max_lon = 250., 320.
        min_lat, max_lat = 5., 25.
    elif maskout == "brazil":
        min_lon, max_lon = 310., 350.
        min_lat, max_lat = -25., -5.
    elif maskout == "namibia":
        min_lon, max_lon = 0., 20.
        min_lat, max_lat = -45., -10.
    else:
        raise NotImplementedError("no such mask type.")
    
    if min_lon > 180 and max_lon > 180:
        min_lon = min_lon - 360
        max_lon = max_lon - 360
    
    add_square(ax,[min_lon,min_lat,max_lon,max_lat],data_crs,**kwargs)

    return ax

def round_to_n(x,n):
    """
    Round to significant digits instead of decimal places
    """
    if x == 0:
        return x
    else:
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
    
def num_lab(x,n):
    """ Converts a number to a string label """
    return str(round_to_n(x,n))

def get_area(lat,lon,mask=None):
    """
    Computes weights for each map cell based off of
    where it is on the globe (Since higher latitude cells will
    appear larger on a map plot then they are in actuality)
    
    Example:
    
    weights = get_area(lat, lon)

    global_mean = np.sum(weights * sst)
    
    will give a proper area-weighted global mean.
    """
    _, Y = np.meshgrid(lon,lat)

    dy = np.deg2rad(np.diff(lat)[0])
    dx = np.deg2rad(np.diff(lon)[0]) * np.cos(np.deg2rad(Y))

    area = dy * dx

    if mask is not None:
        area = area * mask
        area[np.isnan(area)] = 0.

    return area/np.sum(area)