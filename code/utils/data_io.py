"""
Data loading and saving utilities for netCDF climate data.
"""

import xarray as xr
from pathlib import Path
from typing import Union, List, Optional


def load_dataset(
    filepath: Union[str, Path],
    engine: str = "netcdf4",
    chunks: Optional[dict] = None,
    decode_times: bool = True,
    use_cftime: bool = True
) -> xr.Dataset:
    """
    Load a netCDF dataset with standard options.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the netCDF file
    engine : str, default "netcdf4"
        Backend engine for reading
    chunks : dict, optional
        Chunking specification for dask
    decode_times : bool, default True
        Whether to decode time coordinates
    use_cftime : bool, default True
        Whether to use cftime for non-standard calendars
        
    Returns
    -------
    xr.Dataset
        Loaded dataset
    """
    return xr.open_dataset(
        filepath,
        engine=engine,
        chunks=chunks,
        decode_times=decode_times,
        use_cftime=use_cftime
    )


def select_variable(
    ds: xr.Dataset,
    preferred_names: List[str]
) -> str:
    """
    Select a variable from a dataset, trying preferred names first.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to select from
    preferred_names : list of str
        Variable names to try in order of preference
        
    Returns
    -------
    str
        Name of the selected variable
        
    Raises
    ------
    ValueError
        If no suitable variable is found
    """
    # Try preferred names first
    for name in preferred_names:
        if name in ds.data_vars:
            return name
    
    # Fall back to first non-bounds variable
    non_bnds = [name for name in ds.data_vars if not name.endswith("_bnds")]
    if not non_bnds:
        raise ValueError("No usable data variable found in dataset")
    
    return non_bnds[0]

def save_dataset(dataset: xr.Dataset, filename: Union[str, Path]) -> None:
    """
    Save a dataset to a netCDF file.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to save
    filename : str or Path
        Path to the output netCDF file
    """
    dataset.to_netcdf(filename)
    print(f"Saved dataset to {filename}")

