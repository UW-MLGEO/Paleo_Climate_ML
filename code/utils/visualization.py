"""
Visualization utilities for climate data.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import rioxarray as rxr

from typing import Optional, Tuple, Any


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



def plot_time_series(
    data: xr.Dataset | xr.DataArray,
    prefer_var: str = "tas",
    figsize: Tuple[float, float] = (14, 5),
    color: str = "darkblue",
    linewidth: float = 1.5,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a spatial-mean time series for a primary variable in an xarray object.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Input dataset or data array to plot
    prefer_var : str, default "tas"
        Preferred variable name if present in a dataset
    figsize : tuple of float, default (14, 5)
        Figure size
    color : str, default "darkblue"
        Line color
    linewidth : float, default 1.5
        Line width
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if isinstance(data, xr.Dataset):
        if prefer_var in data.data_vars:
            var_name = prefer_var
        else:
            non_bnds = [v for v in data.data_vars if not v.endswith("_bnds")]
            var_name = non_bnds[0] if non_bnds else list(data.data_vars)[0]
        da = data[var_name]
    else:
        da = data
        var_name = da.name or "variable"

    if "time" in da.coords:
        try:
            da = da.assign_coords(time=da.indexes["time"].to_datetimeindex())
        except Exception:
            pass

    if "time" not in da.dims:
        raise ValueError("Input data must include a 'time' dimension for plotting.")

    spatial_dims = [d for d in da.dims if d != "time"]
    time_series = da.mean(spatial_dims) if spatial_dims else da

    units = da.attrs.get("units", "")
    ylabel = f"{var_name} ({units})" if units else var_name
    title = f"Global Mean {var_name} Time Series"

    fig, ax = plot_time_series(
        data=time_series,
        title=title,
        ylabel=ylabel,
        xlabel="Year",
        figsize=figsize,
        color=color,
        linewidth=linewidth,
        ax=ax
    )

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


