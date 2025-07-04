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





