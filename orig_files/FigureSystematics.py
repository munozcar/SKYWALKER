import numpy as np
import batman
import corner
import argparse
import exoparams
import json
import matplotlib
import matplotlib.pyplot as plt
from os import environ
from pandas import DataFrame
from scipy import special
from sklearn.externals import joblib
from functools import partial
from lmfit import Parameters, Minimizer, report_errors
from scipy import spatial
from statsmodels.robust import scale
from time import time
# SKYWALKER methods and assisting routines:

import skywalker
import utils
import models
import bliss
import krdata as kr
import pld
# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0

# Arguments
method = 'bliss'
dataDir = '../GJ1214b_TSO/data/cat_all.joblib.save'
#dataDir = '../GJ1214b_TSO/data/GJ1214b_group0_ExoplanetTSO_transit1.joblib.save'
planet_name = '../GJ1214b_TSO/data/gj1214b_planet_params.json'
x_bin_size = 0.1
y_bin_size = 0.1

transit_type='primary'
x_sigma_range = 4
y_sigma_range = 4
f_sigma_range = 4
init_u1, init_u2, init_u3, init_u4, init_fpfs = None, None, None, None, None

if planet_name[-5:] == '.json':
    with open(planet_name, 'r') as file_in:
        planet_json = json.load(file_in)
    init_period = planet_json['period']
    init_t0     = planet_json['t0']
    init_aprs   = planet_json['aprs']
    init_inc    = planet_json['inc']

    if 'tdepth' in planet_json.keys():
        init_tdepth   = planet_json['tdepth']
    elif 'rprs' in planet_json.keys():
        init_tdepth   = planet_json['rprs']**2
    elif 'rp' in planet_json.keys():
        init_tdepth   = planet_json['rp']**2
    else:
        raise ValueError("Eitehr `tdepth` or `rprs` or `rp` (in relative units) \
                            must be included in {}".format(planet_name))

    init_fpfs   = planet_json['fpfs'] if 'fpfs' in planet_json.keys() else 500 / ppm
    init_ecc    = planet_json['ecc']
    init_omega  = planet_json['omega']
    init_u1     = planet_json['u1'] if 'u1' in planet_json.keys() else None
    init_u2     = planet_json['u2'] if 'u2' in planet_json.keys() else None
    init_u3     = planet_json['u3'] if 'u3' in planet_json.keys() else None
    init_u4     = planet_json['u4'] if 'u4' in planet_json.keys() else None

    if 'planet name' in planet_json.keys():
        planet_name = planet_json['planet name']
    else:
        # Assume the json file name is the planet name
        #   This is a bad assumption; but it is one that users will understand
        print("'planet name' is not inlcude in {};".format(planet_name), end=" ")
        planet_name = planet_name.split('.json')[0]
        print(" assuming the 'planet name' is {}".format(planet_name))
else:
    init_period, init_t0, init_aprs, init_inc, init_tdepth, init_ecc, init_omega = exoparams_to_lmfit_params(planet_name)

init_fpfs = 500 / ppm if init_fpfs is None else init_fpfs
init_u1   = 0.1 if init_u1 is None else init_u1
init_u2   = 0.0 if init_u2 is None else init_u2
init_u3   = 0.0 if init_u3 is None else init_u3
init_u4   = 0.0 if init_u4 is None else init_u4

print('Acquiring Data')

fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths, knots, nearIndices, keep_inds, ind_kdtree, gw_kdtree = utils.setup_inputs_from_file(dataDir, x_bin_size=x_bin_size, y_bin_size=y_bin_size, xSigmaRange=x_sigma_range, ySigmaRange=y_sigma_range, fSigmaRange=f_sigma_range, flux_key='phots', time_key='times', flux_err_key='noise', eff_width_key = 'npix', pld_coeff_key = 'pld', ycenter_key='ycenters', xcenter_key='xcenters', ywidth_key='ywidths', xwidth_key='xwidths', method=method)

# Only use the valid values
fluxes = fluxes[keep_inds]
times = times[keep_inds]
flux_errs = flux_errs[keep_inds]
npix = npix[keep_inds]
xcenters = xcenters[keep_inds]
ycenters = ycenters[keep_inds]
xwidths = xwidths[keep_inds]
ywidths = ywidths[keep_inds]

# Normalize pld pixel values
if method.lower() == 'pld':
    pldints = [pix[keep_inds] for pix in pld_intensities]
    pld_intensities = pld.normalize_pld(np.array(pldints))

# Normalize fluxes and flux errors around 1
flux_errs = flux_errs/np.median(fluxes)
fluxes = fluxes/np.median(fluxes)


transit_indices = [[0,len(times)]]

print('Fixing Time Stamps')
len_init_t0 = len(str(int(init_t0)))
len_times = len(str(int(times.mean())))

# Check if `init_t0` is in JD or MJD
if len_init_t0 == 7 and len_times != 7:
    if len_times == 5:
        init_t0 = init_t0 - 2400000.5
    elif len_times == 4:
        init_t0 = init_t0 - 2450000.5
    else:
        raise ValueError('The `init_t0` is {} and `times.mean()` is {}'.format(int(init_t0), int(times.mean())))

# Check if `init_t0` is in MJD or Simplified-MJD
if len(str(int(init_t0))) > len(str(int(times.mean()))): init_t0 = init_t0 - 50000

print('Initializing Parameters')
initialParams = Parameters()
initialParams.add_many(
    ('period'   , init_period, False),
    ('deltaTc'  , -0.00005   , True  ,-0.05, 0.05),
    ('deltaEc'  , 0.00005    , True  ,-0.05, 0.05),
    ('inc'      , init_inc   , False, 80.0, 90.),
    ('aprs'     , init_aprs  , False, 0.0, 100.),
    ('tdepth'   , init_tdepth, True , 0.0, 0.3 ),
    ('edepth'   , init_fpfs  , True, 0.0, 0.1),
    ('ecc'      , init_ecc   , False, 0.0, 1.0 ),
    ('omega'    , init_omega , False, 0.0, 1.0 ),
    ('u1'       , init_u1    , True , 0.0, 1.0 ),
    ('u2'       , init_u2    , True,  0.0, 1.0 ),
    ('tCenter'  , init_t0    , False),
    ('intcept', 1.0, True),
    ('slope', 0.0, True),
    ('crvtur', 0.0, False))

phase = utils.compute_phase(times, init_t0, init_period)

partial_residuals  = partial(skywalker.residuals_func,
                             times       = times,
                             xcenters    = xcenters,
                             ycenters    = ycenters,
                             fluxes      = fluxes,
                             flux_errs   = flux_errs,
                             knots       = knots,
                             nearIndices = nearIndices,
                             keep_inds   = keep_inds,
                             ind_kdtree  = ind_kdtree,
                             pld_intensities = pld_intensities,
                             method = method.lower(),
                             gw_kdtree = gw_kdtree,
                             transit_indices=transit_indices,
                             x_bin_size = x_bin_size,
                             y_bin_size = y_bin_size)

print('Fitting the Model')
# Setup up the call to minimize the residuals (i.e. ChiSq)
mle0  = Minimizer(partial_residuals, initialParams)
start = time()
fitResult = mle0.leastsq() # Go-Go Gadget Fitting Routine
print("LMFIT operation took {} seconds".format(time()-start))

report_errors(fitResult.params)

print('Establishing the Best Fit Solution')

bf_model_set = skywalker.generate_best_fit_solution(fitResult.params, times=times, xcenters=xcenters, ycenters=ycenters, fluxes=fluxes, knots=knots, keep_inds=keep_inds, method=method, nearIndices=nearIndices, ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree, pld_intensities=pld_intensities, x_bin_size=x_bin_size, y_bin_size=y_bin_size, transit_indices=transit_indices)

bf_full_model = bf_model_set['full_model']
bf_line_model = bf_model_set['line_model']
bf_transit_model = bf_model_set['transit_model']
bf_sensitivity_map = bf_model_set['sensitivity_map']

residuals = fluxes/bf_transit_model
rem = np.array([True if r < 1.05 and r > 0.95 else False for r in residuals])

#ax = plt.subplot2grid((2,2),(0,1), rowspan=2)
#plt.scatter(xcenters[rem], ycenters[rem], c=residuals[rem], s=2, marker='o', cmap='jet')
#plt.xlabel('x-centroid', fontsize=16)
#plt.ylabel('y-centroid', fontsize=16)
#plt.axhline(y=15.5, c='black')
#plt.axvline(x=15.5, c='black')
#plt.axhline(y=14.5, c='black')
#plt.axvline(x=14.5, c='black')
#plt.axvline(x=13.5, c='black')
#plt.colorbar(orientation="vertical")
#rem = np.array([True if r < 1.02 and r > 0.97 else False for r in fluxes])
#ax2 = plt.subplot2grid((2,1),(0,0))
#plt.scatter(times[rem], xcenters[rem], c='black', s=0.2, marker='.', cmap='jet')
#plt.ylabel('x-centroid', fontsize=16)
#
#
#ax3 = plt.subplot2grid((2,1),(1,0),sharex=ax2)
#plt.scatter(times[rem], ycenters[rem], c='black', s=0.2, marker='.', cmap='jet')
#plt.xlabel('Time (days)', fontsize=16)
#plt.ylabel('y-centroid', fontsize=16)
#plt.show()
#
times = utils.bin_data(times, 50)
xcenters = utils.bin_data(xcenters, 50)
ycenters = utils.bin_data(ycenters, 50)
fluxes = utils.bin_data(fluxes, 50)
flux_errs = utils.bin_data(flux_errs, 50)
flux_errs = flux_errs/50
bf_full_model = utils.bin_data(bf_full_model, 50)
bf_transit_model = utils.bin_data(bf_transit_model, 50)
phase = utils.bin_data(phase, 50)


plt.subplot2grid((2,1),(0,0))
plt.scatter(times, xcenters, s=0.5, c='blue')
plt.ylim(14,16)
plt.ylabel('x-centroid', fontsize=16)
plt.subplot2grid((2,1),(1,0))
plt.scatter(times, ycenters, s=0.5, c='blue')
plt.ylim(14,16)
plt.ylabel('y-centroid', fontsize=16)
plt.xlabel('Time (days)', fontsize=16)
plt.show()


plt.subplot2grid ((2,1),(0,0))
plt.plot(times, fluxes,  linewidth=1, color='black', alpha=1)
plt.plot(times, bf_full_model, color='red', linewidth=1)
plt.scatter(times, fluxes, color='black', s=1)
plt.scatter(times, bf_full_model, color='red', s=1)
plt.subplot2grid ((2,1),(1,0))
plt.plot(times, fluxes-bf_full_model, linewidth=1, color='blue')
plt.scatter(times, fluxes-bf_full_model, s=1, color='blue')
plt.show()
