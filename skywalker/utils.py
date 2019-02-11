from scipy import spatial
from sklearn.externals import joblib
from pylab import *;

from tqdm import tqdm

from . import bliss
from . import krdata as kr

import math
# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0

def extractData(file, flux_key='phots', time_key='times', flux_err_key='noise', 
                eff_width_key = 'npix', pld_coeff_key = 'pld', ycenter_key='ycenters', 
                xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths'):
    
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

def setup_inputs_from_file(dataDir, x_bin_size=0.1, y_bin_size=0.1, xSigmaRange=4, 
                           ySigmaRange=4, fSigmaRange=4, flux_key='phots', 
                           time_key='times', flux_err_key='noise', eff_width_key = 'npix', 
                           pld_coeff_key = 'pld', ycenter_key='ycenters', 
                           xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths', 
                           method=None):
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
    assert ('bliss' in method.lower() or  'krdata' in method.lower() or 'pld' in method.lower()), "No valid method selected."
    
    print('Setting up inputs for {}.'.format(method))
    
    fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths = extractData(file = dataDir, 
                                                                                                        flux_key = flux_key, 
                                                                                                        time_key = time_key, 
                                                                                                        flux_err_key = flux_err_key, 
                                                                                                        eff_width_key = eff_width_key, 
                                                                                                        pld_coeff_key = pld_coeff_key, 
                                                                                                        ycenter_key = ycenter_key, 
                                                                                                        xcenter_key = xcenter_key, 
                                                                                                        ywidth_key = ywidth_key, 
                                                                                                        xwidth_key = xwidth_key)
    
    # fluxes is None by default for now...
    keep_inds = removeOutliers(xcenters, ycenters, fluxes=None, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange, f_sigma_cutoff=fSigmaRange)
    if 'bliss' in method.lower():
        print('Setting up BLISS')
        knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], x_bin_size, y_bin_size)
        
        print('BLISS will use a total of {} knots'.format(len(knots)))
        knotTree = spatial.cKDTree(knots)
        nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)
        
        ind_kdtree = None
        gw_kdtree = None
    elif 'krdata' in method.lower():
        print('Setting up KRDATA')
        n_nbr = 100
        expansion = 1000

        xpos = xcenters[keep_inds] - np.median(xcenters[keep_inds])
        ypos = (ycenters[keep_inds] - np.median(ycenters[keep_inds]))/0.7
        np0 = sqrt(npix[keep_inds])
        np0 = (np0 - median(np0))
        
        points = np.transpose([xpos, ypos, np0])
        kdtree = spatial.cKDTree(points * expansion)

        ind_kdtree = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]
        gw_kdtree = kr.gaussian_weights_and_nearest_neighbors(  xpos = xpos,
                                                                ypos = ypos,
                                                                npix = np0,
                                                                inds = ind_kdtree )
        
        knots = None
        nearIndices = None
    elif 'pld' in method.lower():
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

def sub_ax_split(x, y, axs, markersize=5, alpha=0.5, lw=0, mew=0, points_per_bins=10, histtype='stepfilled', orientation='horizontal', color='gray', label=None, returnHist=False):
    ax_scat, ax_hist = axs
    ax_scat.plot(x, y, '.', color=color, markersize=markersize, alpha=alpha, mew=0, label=label)
    yhist, xhist, patches = ax_hist.hist(y, bins=y.size//points_per_bins, histtype=histtype, orientation=orientation, color=color, density=False);
    
    if returnHist: return yhist, xhist, patches

def bin_array(arr, uncs = None,  binsize=100, KeepTheChange = False):
    ''' From `spitzer_helper_functions`
        
        Given nSize = size(time), nCols = binsize, nRows = nSize / nCols

            this function reshapes the 1D array into a new 2D array of dimension
                nCols x nRows or binsize x (nSize / nCols)

            after which, the function collapses the array into a new 1D array by taking the median

        Because most input arrays cannot be subdivided into an even number of subarrays of size `binsize`
            we actually first slice the array into a 1D array of size `nRows*binsize`.

            The mean of the remaining elements from the input array is then taken as the final element
                in the output array

    '''
    from numpy import mean, std, sqrt
    
    nSize   = arr.size
    nCols   = int(nSize / binsize)
    nRows   = binsize

    EqSize  = nRows*nCols
    
    useArr  = arr[:EqSize].copy()   # this array can be subdivided evenly
    
    if uncs is not None:
        # weighted mean profile
        useUncs = uncs[:EqSize].copy()   # this array can be subdivided evenly
        binArr  = median((useArr / useUncs).reshape(nCols, nRows).mean(axis=1)/ useUncs.reshape(nCols, nRows))
        stdArr  = median((useArr / useUncs).reshape(nCols, nRows).std(axis=1) / useUncs.reshape(nCols, nRows))
        
        if KeepTheChange:
            SpareArr    = arr[EqSize:].copy()
            SpareUncs   = uncs[EqSize:].copy()

            binTC       = median((SpareArr / SpareUncs)) / median(SpareUncs.reshape(nCols, nRows))
            stdTC       = median((SpareArr / SpareUncs)) / median(SpareUncs.reshape(nCols, nRows))

            binArr  = concatenate((binArr, [binTC]))
            stdArr  = concatenate((stdArr, [stdTC]))
    else:
        # standard mean profile
        binArr  = mean(useArr.reshape(nCols, nRows),axis=1)
        stdArr  = std(useArr.reshape(nCols, nRows),axis=1) / sqrt(nSize)
        
        if KeepTheChange:
            SpareArr    = arr[EqSize:].copy()
            binTC       = median(SpareArr)
            stdTC       = std(SpareArr)

            binArr  = concatenate((binArr, [binTC]))
            stdArr  = concatenate((stdArr, [stdTC]))

    return binArr, stdArr

def rms_vs_binsize(residuals):
    ''' Compute the RMS (std) as a function of the binsize
    
        Inputs
        ------
            
            residuals (ndarray): the array over which to compute the rms
                                    nominally, this array should be approximately "random noise" 
                                    and free of as much red noise as is possible to compute
        Returns
        -------
            rms_v_bs (ndarray): the computed RMS as a function of binsize.
                                    the minimum binsize is "no binning" (binsize=1)
                                    the maximum binsize is the full array (binsize = residuals.size )
            
            bins_arr (ndarry): an array containing the bins sizes over which `rms_v_bs` was computer
    '''
    min_bin = 2
    max_bin = residuals.size
    
    rms_v_bs = np.zeros(residuals.size-1)
    rms_v_bs[0] = residuals.std()
    
    bins_arr = np.ones(max_bin-1)
    
    for k,binsize in tqdm(enumerate(range(min_bin, max_bin)), total=max_bin-min_bin):
        n_points_per_bin = residuals.size // binsize
        res_bin, res_unc = bin_array(residuals, binsize=binsize)
        rms_v_bs[k+1] = res_bin.std() #/ np.sqrt(n_points_per_bin)
        bins_arr[k+1] = binsize
    
    return rms_v_bs / rms_v_bs[0], bins_arr, 

def plot_rms_vs_binsize(model_set, fluxes, model_rms_v_bs=None, bins_arr=None, times=None, color=None, label=None, zorder=None, alpha = None, ax=None):
    ''' Plots the RMS vs Binsize
        
            Inputs
            ------
                
                model_set (dict): output from `skywalker.generate_best_fit_solution`; dictionary of computed models from parameters
    
                fluxes (ndarray): raw photometry by which to compare the computed models
                
                model_rms_v_bs (ndarray or None): (optional) a precomputed output of `rms_vs_binsize`
                
                bins_arr (ndarray or None): (optional) a precomputed output of `rms_vs_binsize`
                
                times (ndarray or None): (optional) the times array to compare timescales -- Does not work yet
                
                label (string or None): (optional) a label for this rms_vs_binsize, when
                
                ax (matplotlib.axes or None): (optional) an axes handle on which to add a new rms_vs_binsize line
            
            Returns
            -------
                
                ax (matplotlib.axes): the axes handle that was used to plot a new line (either generated here or received as input)
    '''
    
    if ax is None: 
        fig, ax = plt.subplots()
    else:
        try:
            ax.lines.pop() # Remove axvline -- will be added again here
            ax.lines.pop() # Remove Gaussian theory model -- will be added again here
        except:
            pass # This is the first plot
    
    transit_duration = np.where(model_set['transit_model'] < model_set['transit_model'].max())[0]
    transit_duration = transit_duration.max() - transit_duration.min()
    
    if model_rms_v_bs is None or bins_arr is None: 
        full_model = model_set['full_model']
        residuals = fluxes - full_model
        model_rms_v_bs, bins_arr = rms_vs_binsize(residuals)
    
    theory_gauss = 1/np.sqrt(bins_arr)
    theory_gauss /= theory_gauss[0]
    
    if label is not None:
        label = label + ' {:.2f}x'.format((model_rms_v_bs / theory_gauss)[transit_duration])
    
    if times is not None: 
        print('This doent work!')
        bins_arr = times[np.int32(bins_arr)]
        transit_duration = times[np.int32(transit_duration)]
    
    ax.loglog(bins_arr, model_rms_v_bs, lw=1, color=color, label=label, zorder=zorder, alpha=alpha);
    ax.loglog(bins_arr, theory_gauss, color='black', ls='--', lw=1, label='Gaussian Residuals');
    ax.axvline(transit_duration, color='darkgrey', ls='--', lw=1)
    ax.set_ylim(theory_gauss.min()/1.1,1.1*theory_gauss.max())
    
    ax.legend(loc=0)
    
    return ax

def plot_fit_residuals_physics(times, fluxes, flux_errs, model_set, planet_name='', channel='', staticRad='', 
                                    varRad='', method='', plot_name='', time_stamp='', nSig=3, save_now=False, bin_size=10,
                                    label_kwargs={}, title_kwargs={}, color_cycle = None, hspace=3.0, figsize=None,
                                    markersize=2, alpha=1, lw=0.5, mew=0, returnAx=False, save_dir='', points_per_bins=10):
    
    if 'fontsize' not in title_kwargs.keys(): title_kwargs['fontsize'] = 5
    
    if 'fontsize' not in label_kwargs.keys(): label_kwargs['fontsize'] = 5
    
    if color_cycle is None: color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    print('Establishing the {} Solution'.format(plot_name))
    
    full_model = model_set['full_model']
    line_model = model_set['line_model']
    physical_model = model_set['physical_model']
    sensitivity_map = model_set['sensitivity_map']
    weirdness = model_set['weirdness']
    
    times_binned = bin_data(times, bin_size = bin_size)
    fluxes_binned = bin_data(fluxes, bin_size = bin_size)
    flux_errs_binned = bin_data(flux_errs, bin_size = bin_size) / np.sqrt(bin_size)
    
    full_model_binned = bin_data(full_model, bin_size = bin_size)
    physical_model_binned = bin_data(physical_model, bin_size = bin_size)
    line_model_binned = bin_data(line_model, bin_size = bin_size)
    sensitivity_map_binned = bin_data(sensitivity_map, bin_size = bin_size)
    weirdness_binned = bin_data(weirdness, bin_size = bin_size) if type(weirdness) is np.ndarray else 1.0
    
    std_res = np.std(fluxes_binned-full_model_binned)
    
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(20,20), sharex=True)
    # Set up the axes with gridspec
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(6, 6, hspace=hspace, wspace=0.0)
    
    raw_scat = fig.add_subplot(grid[:2, :-1], xticklabels=[])
    raw_hist = fig.add_subplot(grid[:2, -1], xticklabels=[], sharey=raw_scat)
    raw_hist.yaxis.tick_right()
    
    res_scat = fig.add_subplot(grid[2:4, :-1], xticklabels=[])
    res_hist = fig.add_subplot(grid[2:4, -1], xticklabels=[], sharey=res_scat)
    res_hist.yaxis.tick_right()
    
    phy_scat = fig.add_subplot(grid[4:, :-1])
    phy_hist = fig.add_subplot(grid[4:, -1], sharey=phy_scat, xticklabels=[])
    phy_hist.yaxis.tick_right()
    
    # Raw Flux Plots
    sub_ax_split(times_binned - times.mean(), ppm*(full_model_binned-1), [raw_scat, raw_hist], color=color_cycle[0], label='Binned Normalized Flux',
                            markersize=markersize, alpha=alpha, lw=lw, mew=mew, points_per_bins=points_per_bins)
    
    raw_scat.errorbar(times_binned - times.mean(), ppm*(fluxes_binned-1), yerr=ppm*flux_errs_binned, fmt='.', mew=0,
                        color = color_cycle[1], lw=lw, markersize=markersize, label='Full Initial Model', zorder=0, alpha=0.5)
    raw_scat.set_title('Binned Normalized Flux from {} - {}; {}; {}; {}; {}; {}'.format('Init', planet_name, channel, staticRad, 
                                                                                   varRad, method, plot_name), **title_kwargs)
    raw_scat.set_xlabel('Time [days]', **label_kwargs)
    raw_scat.set_ylabel('Normalized Flux [ppm]', **label_kwargs)
    
    # Residual Flux Plots
    yhist, xhist, _ = sub_ax_split(times_binned - times.mean(), ppm*(fluxes_binned-full_model_binned), [res_scat, res_hist], color=color_cycle[4], returnHist=True,
                            markersize=markersize, alpha=alpha, lw=lw, mew=mew, points_per_bins=points_per_bins)
    
    res_scat.set_title('Residuals from {} - {}; {}; {}; {}; {}; {}'.format('Init', planet_name, channel, staticRad, varRad, method, plot_name), **title_kwargs)
    res_scat.set_xlabel('Time [days]', **label_kwargs)
    res_scat.set_ylabel('Residuals [ppm]', **label_kwargs)
    res_scat.set_ylim(-ppm*nSig*std_res, ppm*nSig*std_res)
    
    res_hist.annotate('{:.0f} ppm'.format(np.ceil(ppm*std_res)), (0.1*yhist.max(), np.ceil( 1.1*ppm*std_res)), fontsize=label_kwargs['fontsize'], color='indigo')
    res_hist.annotate('{:.0f} ppm'.format(np.ceil(ppm*std_res)), (0.1*yhist.max(), np.ceil(-1.5*ppm*std_res)), fontsize=label_kwargs['fontsize'], color='indigo')
    res_hist.axhline( ppm*std_res, ls='--', color='indigo')
    res_hist.axhline(-ppm*std_res, ls='--', color='indigo')
    
    # Physics Only Plots
    sub_ax_split(times_binned - times.mean(), ppm*(fluxes_binned / sensitivity_map_binned / line_model_binned / weirdness_binned - 1.0), [phy_scat, phy_hist], color=color_cycle[0],
                            markersize=markersize, alpha=alpha, lw=lw, mew=mew, points_per_bins=points_per_bins)
    
    phy_scat.plot(times_binned - times.mean(), ppm*(physical_model_binned / line_model_binned - 1.0), color=color_cycle[1], zorder=100)
    phy_scat.axhline(0.0, ls='--', color=color_cycle[2])
    phy_scat.set_title('Physics Only from {} - {}; {}; {}; {}; {}; {}'.format('Init', planet_name, channel, staticRad, varRad, method, plot_name), **title_kwargs)
    phy_scat.set_xlabel('Time [days]', **label_kwargs)
    phy_scat.set_ylabel('Residuals [ppm]', **label_kwargs)
    phy_scat.set_ylim(-ppm*nSig*std_res, ppm*nSig*std_res)
    
    for tick in phy_scat.xaxis.get_major_ticks(): tick.label.set_fontsize(label_kwargs['fontsize'])
    
    for tick in raw_scat.yaxis.get_major_ticks(): tick.label1.set_fontsize(label_kwargs['fontsize'])
    for tick in res_scat.yaxis.get_major_ticks(): tick.label1.set_fontsize(label_kwargs['fontsize'])
    for tick in phy_scat.yaxis.get_major_ticks(): tick.label1.set_fontsize(label_kwargs['fontsize'])
    for tick in raw_hist.yaxis.get_major_ticks(): tick.label2.set_fontsize(label_kwargs['fontsize'])
    for tick in res_hist.yaxis.get_major_ticks(): tick.label2.set_fontsize(label_kwargs['fontsize'])
    for tick in phy_hist.yaxis.get_major_ticks(): tick.label2.set_fontsize(label_kwargs['fontsize'])
    
    if save_now: 
        plot_save_name = '{}_{}_{}_{}_{}_{}_fit_residuals_physics_{}.png'.format(planet_name.replace(' b','b'), channel, staticRad, 
                                                                                     varRad, method, plot_name, time_stamp)
        
        print('Saving Initial Fit Residuals Plot to {}'.format(save_dir + plot_save_name))
        fig.savefig(save_dir + plot_save_name)
    
    if returnAx: return raw_scat, raw_hist, res_scat, res_hist, phy_scat, phy_hist