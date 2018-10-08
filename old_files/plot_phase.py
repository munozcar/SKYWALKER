import argparse
from sklearn.externals import joblib
import models
import matplotlib.pyplot as plt
import batman
import numpy as np
import utils

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--spitzer_file', type=str, required=True , help='File storing the lmfit results')
ap.add_argument('-r', '--results_file', type=str, required=True , help='File storing the lmfit results')
args = vars(ap.parse_args())

dataDir = args['results_file']
spitzer = args['spitzer_file']

fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths, knots, nearIndices, keep_inds, ind_kdtree, gw_kdtree = utils.setup_inputs_from_file(spitzer, x_bin_size=0.1, y_bin_size=0.1, xSigmaRange=3, ySigmaRange=3, fSigmaRange=3, flux_key='phots', time_key='times', flux_err_key='noise', eff_width_key = 'npix', pld_coeff_key = 'pld', ycenter_key='ycenters', xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths', method= 'pld')

# Only use the valid values
fluxes = fluxes[keep_inds]
times = times[keep_inds]
flux_errs = flux_errs[keep_inds]
npix = npix[keep_inds]
xcenters = xcenters[keep_inds]
ycenters = ycenters[keep_inds]
xwidths = xwidths[keep_inds]
ywidths = ywidths[keep_inds]

# Normalize fluxes and flux errors around 1
flux_errs = flux_errs/np.median(fluxes)
fluxes = fluxes/np.median(fluxes)


print('Acquiring Data')

results = joblib.load(dataDir)
transit = models.transit_model_func(results.params, times = times, init_t0=results.params['tCenter'], transitType='primary')
eclipse = models.transit_model_func(results.params, times = times, init_t0=results.params['tCenter'], transitType='secondary')

plt.scatter(times, fluxes, color='darkblue', s=0.5, alpha=0.5)
plt.plot(times, transit, color='red', linewidth = 3)

plt.show()
plt.plot(times, transit+(0.1*np.sin(times*2*np.pi/results.params['period'])))
plt.plot(times, transit, color='red', linewidth = 3)

plt.show()
