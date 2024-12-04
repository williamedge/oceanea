import numpy as np
import xarray as xr
from .data_checks import _timedim_check, well_spaced_obj



@xr.register_dataarray_accessor("floatda")
class WSspectral():
    '''
    This class is a wrapper around a well-spaced xarray DataArray object. It is used to
    provide a set of methods to manipulate the data in the DataArray. The
    methods are implemented in the XXXX.
    '''


    # Initial checks upon class creation
    def __init__(self, da):

        # Check if the input object is a DataArray
        if not isinstance(da, xr.DataArray):
            raise(Exception("Input must be an xarray DataArray"))

        # Extract units if possible
        if 'units' in da.attrs:
            self.units = da.attrs['units']
        else:
            self.units = '?'

        # Check if time is a dimension, if so it must be first
        dims = np.array(da.dims)
        _timedim_check(dims)

        # Check if array is well-spaced
        dimcheck = well_spaced_obj(da, verbose=False)
        if not np.all(dimcheck):
            # Raise error and display dimensions that are not well spaced
            raise(Exception("Array dimensions not well spaced: " + \
                            str(dims[~dimcheck])))

        # Store the DataArray object
        self._obj = da


    # Property to access the DataArray object
    @property
    def _da(self):
        return self._obj
    
    # Property to access the dimensions of the DataArray
    @property
    def dims(self):
        return self._obj.dims
    
    # Property to access the dimensions of the DataArray
    # @property





def calc_fundamental(dimvals, safety=1, dt64_units='s', ):
    '''
    Calc fundamental freq approx. of a 1D series
    in cycles per specified unit (defaults to seconds for time)
    '''
    # Check if dimvals is time or float
    if np.issubdtype(dimvals.dtype, np.datetime64):
        # Calc fundamental freq
        funfq = safety/(1 * len(dimvals) * (np.diff(dimvals)[0] / np.timedelta64(1,dt64_units))
                        * (np.timedelta64(1,dt64_units).astype('int')))
    elif np.issubdtype(dimvals.dtype, np.float64) | np.issubdtype(dimvals.dtype, np.int64):
        # Calc fundamental freq
        funfq = safety/(len(dimvals))
    else:
        raise(Exception("\'dimvals\' must be time or float"))
    return funfq


def calc_nyquist(dimvals, safety=2, dt64_units='s'):
    '''
    Calc nyquist approx. in cycles per unit of dimension (defaults to seconds for time)
    dimvals: dimension array (time or float)
    safety: factor to divide by (min. 2, higher for noisey data)
    '''
    # Check if dimvals needs to be squeezed
    if len(dimvals.shape) > 1:
        dimvals = np.squeeze(dimvals)
    if np.issubdtype(dimvals.dtype, np.datetime64):
        return 1/(safety * (np.diff(dimvals)[0] / np.timedelta64(1,dt64_units)))
    elif np.issubdtype(dimvals.dtype, np.float64) | np.issubdtype(dimvals.dtype, np.int64):
        return 1/(safety * np.diff(dimvals)[0])