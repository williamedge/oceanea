import numpy as np



def _timedim_check(dims):
    if 'time' in dims:
        if not dims[0].lower() == 'time':
            raise(Exception("First dimension must be time"))
    return None


def well_spaced_dim(xarray_dim, verbose=True):
    """
    Check if a supplied xarray dimension is evenly spaced
    """
    
    diff = xarray_dim.diff(xarray_dim.dims[0])
    if np.all(diff == diff[0]):
        if verbose:
            print(xarray_dim.dims[0] + ' dimension well spaced')
        return True
    else:
        if verbose:
            print(xarray_dim.dims[0] + ' dimension not well spaced!!')
        return False


def well_spaced_obj(xr_dataobj, verbose=True):
    """
    Check if a supplied xarray dataarray / dataset dimensions are evenly spaced
    """
    check = np.full(len(xr_dataobj.dims), False)
    for i, dim in enumerate(xr_dataobj.dims):
        check[i] = well_spaced_dim(xr_dataobj[dim], verbose=verbose)
    return check