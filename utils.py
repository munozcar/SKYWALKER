from scipy import spatial
from sklearn.externals import joblib
from pylab import *;
import bliss
import krdata as kr
import math
# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0

def extractData(file, flux_key='phots', time_key='times', flux_err_key='noise', eff_width_key = 'npix', pld_coeff_key = 'pld', ycenter_key='ycenters', xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths'):

    group = joblib.load(file)

    fluxes = group[flux_key].flatten()
    times = group[time_key].flatten()
    flux_errs = group[flux_err_key].flatten()
    npix = group[eff_width_key].flatten()
    pld_intensities = group[pld_coeff_key]
    xcenters = group[xcenter_key].flatten()
    ycenters = group[ycenter_key].flatten()
    xwidths = group[xwidth_key].flatten()
    ywidths = group[ywidth_key].flatten()

    return fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths

def nearest(xc, yc, neighbors, tree):
    """
        Args:
        point: list of x, y coordinates of a single center.
        neighbors (int): how many neighbors to look for.
        tree: precomputed spacial.KDtree() of a grid of knots.
        Returns:
        neighbors: array of indices of the nearest neighbors.
    """
    neighbors = tree.query((xc,yc), k=neighbors)
    return neighbors[1]

def removeOutliers(xcenters, ycenters, fluxes=None, x_sigma_cutoff=3, y_sigma_cutoff=4, f_sigma_cutoff=4):
    """
        Args:
        xcenters (nDarray): array of x-coordinates of centers.
        ycenters (nDarray): array of y-coordinates of centers.
        fluxes (list or None): None or array of fluxes associated with the centers. (if fluxes is None, then skip 3rd dimension)
        x_sigma_cutoff (float): how many standard deviatins to accept in x.
        y_sigma_cutoff (float): how many standard deviatins to accept in y.
        f_sigma_cutoff (float): how many standard deviatins to accept in y.
        Returns:
        boolean list: list of indices to keep as inliers
    """
    x_ell = ((xcenters - xcenters.mean())/x_sigma_cutoff)**2. # x-ellipse term
    y_ell = ((ycenters - ycenters.mean())/y_sigma_cutoff)**2. # y-ellipse term
    if fluxes is not None: f_ell = ((fluxes   - fluxes.mean()  )/f_sigma_cutoff)**2. # flux-ellipse term

    return y_ell + x_ell + f_ell < 1 if fluxes is not None else y_ell + x_ell < 1

def setup_inputs_from_file(dataDir, x_bin_size=0.1, y_bin_size=0.1, xSigmaRange=4, ySigmaRange=4, fSigmaRange=4, flux_key='phots', time_key='times', flux_err_key='noise', eff_width_key = 'npix', pld_coeff_key = 'pld', ycenter_key='ycenters', xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths', method=None):

    """
    Description:
        This function takes in the filename of the data (stored with sklearn-joblib), checks the data for outliers, establishes the interpolation grid, computes the nearest neighbours between all data points and that grid, and outputs the necessary values for using BLISS. The 'flux' is assumed to be pure stellar signal -- i.e. no planet. BLISS is expected to be used inside a fitting routine where the transit has been 'divided out'. This example here assumes that there is no transit or eclipse in the light curve data (i.e. `flux` == 'stellar flux'). To use this with a light curve that contains a transit or eclipse, send the "residuals" to BLISS: - i.e. `flux = system_flux / transit_model`
        Written by C.Munoz-Romero 07-05-18
        Edited by J.Fraine 07-06-18
    Args:
        dataDir (str): the directory location for the joblib file containing the x,y,flux information.
        x_bin_size (float): distance in x-dimension to space interpolation grid
        y_bin_size (float): distance in y-dimension to space interpolation grid
        xSigmaRange (float): relative distance in gaussian sigma space to reject x-outliers
        ySigmaRange (float): relative distance in gaussian sigma space to reject y-outliers
    Returns:
        xcenters (nDarray): X positions for centering analysis
        ycenters (nDarray): Y positions for centering analysis
        fluxes (nDarray): photon counts from raw data
        flux_err (nDarray): photon uncertainties
        knots (nDarray): locations and initial flux values (weights) for interpolation grid.
        nearIndices (nDarray): nearest neighbour indices per point for location
        of nearest knots keep_inds (list): list of indicies to keep within the thresholds set.
    """
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "No valid method selected."
    print('Setting up inputs for {}.'.format(method))

    fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths = extractData(dataDir, flux_key=flux_key, time_key=time_key, flux_err_key=flux_err_key, eff_width_key = eff_width_key, pld_coeff_key = pld_coeff_key, ycenter_key=ycenter_key, xcenter_key=xcenter_key, ywidth_key=ywidth_key, xwidth_key=xwidth_key)
    # fluxes is none by default for now...
    keep_inds = removeOutliers(xcenters, ycenters, fluxes=None, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange, f_sigma_cutoff=fSigmaRange)
    if method.lower() == 'bliss':
        print('Setting up BLISS')
        knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], x_bin_size, y_bin_size)
        print('BLISS will use a total of {} knots'.format(len(knots)))
        knotTree = spatial.cKDTree(knots)
        nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)
        ind_kdtree = None
        gw_kdtree = None
    elif method.lower() == 'krdata':
        print('Setting up KRDATA')
        n_nbr   = 100
        expansion = 1000

        xpos = xcenters[keep_inds] - np.median(xcenters[keep_inds])
        ypos = (ycenters[keep_inds] - np.median(ycenters[keep_inds]))/0.7
        np0 = sqrt(npix[keep_inds])
        np0 = (np0 - median(np0))

        points  = np.transpose([xpos, ypos, np0])
        kdtree  = spatial.cKDTree(points * expansion)

        ind_kdtree  = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]
        gw_kdtree = kr.gaussian_weights_and_nearest_neighbors(  xpos=xpos,
                                                                ypos=ypos,
                                                                npix=np0,
                                                                inds=ind_kdtree )
        knots = None
        nearIndices = None
    elif method.lower() == 'pld':
        print('Using PLD')
        ind_kdtree = None
        gw_kdtree = None
        knots = None
        nearIndices = None

    return fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths, knots, nearIndices, keep_inds, ind_kdtree, gw_kdtree

def exoparams_to_lmfit_params(planet_name):
    '''
        Args:
        planet_name
        Returns:
        Parameters (Period, Transit center, ApRs, Inclinaiton, Transit depth, Eccentricity, Omega) for the specified planet taken from the exoparams module. (not tested yet)
    '''
    ep_params   = exoparams.PlanetParams(planet_name)
    iApRs       = ep_params.ar.value
    iEcc        = ep_params.ecc.value
    iInc        = ep_params.i.value
    iPeriod     = ep_params.per.value
    iTCenter    = ep_params.tt.value
    iTdepth     = ep_params.depth.value
    iOmega      = ep_params.om.value

    return iPeriod, iTCenter, iApRs, iInc, iTdepth, iEcc, iOmega

def compute_phase(times, t0, period):
    phase = ((times - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    return phase

def bin_data(data, bin_size):
    binned = []
    init = 0
    num_bins = int(len(data)/bin_size)
    for i in np.array(range(num_bins))+1:
        binned.append(np.mean(data[init:i*bin_size]))
        init = i*bin_size
    return np.array(binned)
