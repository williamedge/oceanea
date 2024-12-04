import numpy as np
import xarray as xr
from .data_checks import _timedim_check, well_spaced_obj

from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic_dd
from tqdm import tqdm

@xr.register_dataarray_accessor("floatda")
class Variogram_data():
    '''
    This class is a wrapper around an irregularly spaced xarray DataArray object. It is used to
    provide a set of methods to manipulate the data in the DataArray. The
    methods are implemented directly below.
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
    
    #### Analysis methods ####
    def semivar_ND(self, bins=None, marginal=None, raverage=None, maxchunk=1e4, mode='full',\
                   progress_bar=False, verbose=True, bin_tol=0.34, bin_no=30):
        '''
        Calculate the semivariogram for an nD array of data using scipy binned_statistic_dd
        '''
        # Get the data as a numpy array
        X = np.column_stack([self._obj[c].values for c in self._obj.coords])
        Z = self._obj.values
        
        # Calculate the semivariogram
        bin_centers, results_all , counts_all = semivar_nd(X, Z, bins=bins, marginal=marginal, raverage=raverage, maxchunk=maxchunk, mode=mode,\
                                                           progress_bar=progress_bar, verbose=verbose, bin_tol=bin_tol, bin_no=bin_no)
        return bin_centers, results_all , counts_all



#### Utility funcs ####

def auto_bins(coords, tolerance=0.34, bin_no=30):
    # Generate bins for each column in coords
    bin_edges = [np.linspace(0, (np.max(c) - np.min(c))*tolerance*0.999, bin_no) for c in coords.T]
    return bin_edges
    

def get_nbins(bin_edges):   
    # Get the bin centers - should this be geometric mean for log bins????
    bin_center = [0.5*(be[:-1] + be[1:]) for be in bin_edges]
    # # Make an array if all lengths are equal
    # if len(set([len(bc) for bc in bin_center])) == 1:
    #     bin_center = np.array(bin_center)
    return bin_center


def check_bins(data_coords, bin_edges, tolerance=0.34, radial=None, verbose=True):
    if radial is not None:
        assert len(radial) > 1, 'Must provide at least two columns to calculate radial semivariogram'
        assert len(bin_edges) == data_coords.shape[1]-len(radial)+1,\
            'Must provide bins for each output dimension for radial semivariogram'
        #
        ##### Add a bin check
    else:   
        # Check there is one bin edge OR one bin edge per dimension
        if len(bin_edges) != 1 and len(bin_edges) != data_coords.shape[1]:
            raise ValueError('Invalid number of bin edges - must be one (same bins for all dimensions) or one per dimension')
        
        # Check if any bins are longer than tolerated
        if verbose:
            if np.any([np.max(b) for b in bin_edges] > np.max(data_coords, axis=0)*tolerance):
                max_axis = np.where([np.max(b) for b in bin_edges] > np.max(data_coords, axis=0)*tolerance)[0]
                print('Warning: Maximum bin size is larger than a third the maximum distance between data points for axis: ', max_axis)


def set_chunking(data, mode='full', maxchunk=1e4, verbose=True):
    # Check if splitting is required
    if len(data) > maxchunk:
        if verbose:
            print('Warning: Data size is larger than maxchunk, splitting data into chunks')
        if mode == 'full':
            split = 1
        elif mode == 'fast':
            split = int(maxchunk)
        else:
            raise ValueError('Invalid mode')
    else:
        split = len(data)
        # Override mode in this case
        mode = 'fast'
    return split, mode


def check_variogram_count(var_counts, count_tol=30, verbose=True):
    if verbose:
        if len(np.argwhere(var_counts < count_tol)) > 0:
            print('Warning: Bins with less than 30 pairs: ', np.argwhere(var_counts < count_tol))


def calculate_pdist_bycolumn(X):
    # Calculate the distance matrix for each column in X
    return np.column_stack([pdist(X[:,ii][:,None]) for ii in range(X.shape[1])])


def compute_semivariogram(X_diff, Z_diff, bins):
    'Calculate the semivariogram for an nD array of data using scipy binned_statistic_dd'
    if Z_diff.ndim > 1:
        Z_diff = Z_diff.flatten()
    bin_result, _, _ = binned_statistic_dd(X_diff, 0.5*Z_diff**2, bins=bins,\
                                           statistic='sum', expand_binnumbers=True)
    
    # Compute the number of pairs in each bin - always need to do this now
    bin_count, _, _ = binned_statistic_dd(X_diff, Z_diff, bins=bins,\
                                          statistic='count', expand_binnumbers=True)
    return bin_result, bin_count



##### Var funcs

def semivar_nd(X, Z, bins=None, marginal=None, raverage=None, cutdim=None, maxchunk=1e4, mode='full',\
               progress_bar=False, verbose=True, bin_tol=0.34, bin_no=30, margexcl=None, margtol=None):
    'Calculate the full semivariogram for an nD array of data'
    # Dimension handling and auto-bins
    if bins is None:
        bins = auto_bins(X, tolerance=bin_tol, bin_no=bin_no)
    elif (bins is not None) & ((X.ndim == 1) | (len(bins) > 10)):
        bins = [bins]
    if Z.ndim == 1:
        Z = Z[:,None]
    else:
        raise ValueError('Data values must be 1D')
    assert X.shape[0] == Z.shape[0], 'Data coords and values must have the same length'
    if raverage is not None:
        assert mode=='full', 'Radial averaging only done with full mode (for now)' # This gets overwritten later if size is small
    if margtol is None:
        margtol = bins[0][1] - bins[0][0]

    # Some warnings - max bin size first
    if verbose:
        check_bins(X, bins, tolerance=bin_tol, radial=raverage, verbose=verbose)
    bmax = [np.max(b) for b in bins]

    # Chunking stuff - we will run into problems when 1D array gets too long (maybe billions?)
    split, mode = set_chunking(Z, mode=mode, maxchunk=maxchunk, verbose=verbose)
        
    # Break data into chunks of length split
    for i in tqdm(range(0, Z.size, split), disable=not progress_bar):
        # Fast mode does not pick up pairs in adjacent chunks
        if mode == 'fast':
            if raverage is not None:
                # Get new collapsed data
                # collapsed_dist = np.sqrt(np.sum((X[i:,raverage] - X[i,raverage])**2, axis=1))
                # revcols = list(set(range(X.shape[1])) - set(raverage))
                # X_new = np.column_stack((collapsed_dist, X[i:,revcols]))  ##### This fucks up when dimensions reordered maybe
                # X_diff = calculate_pdist_bycolumn(X_new[i:i+split])
                X_diff = pdist(X[i:i+split])
            else:
                # Get dimension differences
                X_diff = calculate_pdist_bycolumn(X[i:i+split])
            # Get value differences
            val_dif = pdist(Z[i:i+split])
            
        # Full mode picks up all pairs in dataset - limited by memory in 1D
        elif mode == 'full':
            if i < Z.size:
                # Remove extra data
                if cutdim is not None:
                    if marginal is not None:
                        dcut = np.where(X[i+1:,cutdim] > X[i+1,cutdim] + np.max(bins[0]))[0]
                    elif raverage is not None:
                        dcut = np.where(X[i+1:,cutdim] > X[i+1,cutdim] + np.max(bmax))[0]
                    else:
                        dcut = np.where(X[i+1:,cutdim] > X[i+1,cutdim] + np.max(bins[cutdim]))[0]

                    if len(dcut)>0:
                        # Get index where the next point is too far away
                        dc = i + 1 + dcut[0]
                    else:
                        dc = Z.size
                        cutdim = None
                else:
                    dc = Z.size

                # Get dimension differences
                if marginal is not None:
                    X_diff = np.abs(X[i+1:dc,marginal] - X[i,marginal])
                elif raverage is not None:
                    # Get new collapsed data
                    collapsed_dist = np.sqrt(np.sum((X[i:,raverage] - X[i,raverage])**2, axis=1))
                    revcols = list(set(range(X.shape[1])) - set(raverage))
                    X_new = np.column_stack((collapsed_dist, X[i:,revcols]))  ##### This fucks up when dimensions reordered maybe
                    X_diff = np.abs(X_new[1:dc-i] - X_new[0])
                else:
                    X_diff = np.abs(X[i+1:dc] - X[i])
                # Get value differences
                val_dif = np.abs(Z[i+1:dc] - Z[i])

        if marginal is None:
            # Mask out pairs that are too far away (compact)
            if X_diff.ndim > 1:
                mask = ~np.any(X_diff > np.max(bmax), axis=1)
            else:
                mask = ~(X_diff > np.max(bmax))
        elif margexcl is None:
            # Mask out pairs that have euclidean distance too far
            A_diff = np.sqrt(np.sum((X[i+1:dc] - X[i])**2, axis=1))
            bin_idx = np.digitize(X_diff, bins[0])
            bin_ida = np.digitize(A_diff, bins[0])
            mask = np.where(bin_idx==bin_ida)[0]
        elif margexcl is not None:
            revcols = list(set(range(X.shape[1])) - set([marginal]))
            X_off = np.abs(X[i+1:dc,marginal][:,None] - X[i+1:dc,revcols])
            mask = np.all(X_off < margtol, axis=1)

        # Calculate the semivariogram
        if len(val_dif[mask]) > 0:
            bin_result, bin_count = compute_semivariogram(X_diff[mask], val_dif[mask], bins)
                        
        # Compile the results - first check if summation vars exist
        if ('results_all' not in locals()) & (len(val_dif[mask]) > 0):
            results_all = bin_result
            counts_all = bin_count
        elif len(val_dif[mask]) > 0:
            results_all += bin_result
            counts_all += bin_count
        
    # Get the bin centers - should this be geometric mean for log bins????
    bin_centers = get_nbins(bins)

    check_variogram_count(counts_all, verbose=verbose)

    # Make it the mean
    results_all = results_all / counts_all

    return bin_centers, results_all , counts_all

