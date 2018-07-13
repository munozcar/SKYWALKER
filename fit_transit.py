from scipy import special
import corner
import argparse
import batman
import exoparams
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from functools import partial
from lmfit import Parameters, Minimizer, report_errors
from os import environ
from scipy import spatial
from statsmodels.robust import scale
from time import time
from bokeh.io       import output_notebook, show
from bokeh.plotting import figure
from bokeh.models   import Span
from bokeh.layouts  import gridplot
from pandas import DataFrame
y,x = 0,1
ppm = 1e6
import matplotlib
# ****** IMPORT METHODS *******
import BLISS as bliss
import KRdata as kr
import PLD as pld


matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=10)
def setup_BLISS_inputs_from_file(dataDir, xBinSize=0.01, yBinSize=0.01,
                                 xSigmaRange=4, ySigmaRange=4, fSigmaRange=4):
    """This function takes in the filename of the data (stored with sklearn-joblib),
        checks the data for outliers, establishes the interpolation grid,
        computes the nearest neighbours between all data points and that grid,
        and outputs the necessary values for using BLISS

        The `flux` is assumed to be pure stellar signal -- i.e. no planet.
        BLISS is expected to be used inside a fitting routine where the transit has been `divided out`.
        This example here assumes that there is no transit or eclipse in the light curve data (i.e. `flux` == 'stellar flux').
        To use this with a light curve that contains a transit or eclipse, send the "residuals" to BLISS:
            - i.e. `flux = system_flux / transit_model`

    Written by C.Munoz 07-05-18
    Edited by J.Fraine 07-06-18
    Args:
        dataDir (str): the directory location for the joblib file containing the x,y,flux information

        xBinSize (float): distance in x-dimension to space interpolation grid
        yBinSize (float): distance in y-dimension to space interpolation grid
        xSigmaRange (float): relative distance in gaussian sigma space to reject x-outliers
        ySigmaRange (float): relative distance in gaussian sigma space to reject y-outliers
    Returns:
        xcenters (nDarray): X positions for centering analysis
        ycenters (nDarray): Y positions for centering analysis
        fluxes (nDarray): normalized photon counts from raw data
        flux_err (nDarray): normalized photon uncertainties
        knots (nDarray): locations and initial flux values (weights) for polation grid
        nearIndices (nDarray): nearest neighbour indices per point for location of nearest knots
        keep_inds (list): list of indicies to keep within the thresholds set

    """
    times, xcenters, ycenters, fluxes, flux_errs = bliss.extractData(dataDir)

    keep_inds = bliss.removeOutliers(xcenters, ycenters, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange)
    #, fSigmaRange)

    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)

    return times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds

def setup_BLISS_inputs_from_file_custom(dataDir, xBinSize=0.01, yBinSize=0.01, xSigmaRange=4, ySigmaRange=4, fSigmaRange=4):
    times, xcenters, ycenters, fluxes, flux_errs = bliss.extractDatacustom(dataDir)

    keep_inds = bliss.removeOutliers(xcenters, ycenters, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange)
        #, fSigmaRange)

    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)

    return times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds


def deltaphase_eclipse(ecc, omega):
    return 0.5*( 1 + (4. / pi) * ecc * cos(omega))

def transit_model_func(model_params, times, ldtype='quadratic', transitType='primary'):
    # Transit Parameters
    u1      = model_params['u1'].value
    u2      = model_params['u2'].value

    if 'edepth' in model_params.keys() and model_params['edepth'] > 0:
        if 'ecc' in model_params.keys() and 'omega' in model_params.keys() and model_params['ecc'] > 0:
            delta_phase = deltaphase_eclipse(model_params['ecc'], model_params['omega'])
        else:
            delta_phase = 0.5

        t_secondary = model_params['tCenter'] + model_params['period']*delta_phase

    else:
        model_params.add('edepth', 0.0, False)

    rprs  = np.sqrt(model_params['tdepth'].value)

    bm_params           = batman.TransitParams() # object to store transit parameters

    bm_params.per       = model_params['period'].value   # orbital period
    bm_params.t0        = model_params['tCenter'].value  # time of inferior conjunction
    bm_params.inc       = model_params['inc'].value      # inclunaition in degrees
    bm_params.a         = model_params['aprs'].value     # semi-major axis (in units of stellar radii)
    bm_params.rp        = rprs     # planet radius (in units of stellar radii)
    bm_params.fp        = model_params['edepth'].value   # planet radius (in units of stellar radii)
    bm_params.ecc       = model_params['ecc'].value      # eccentricity
    bm_params.w         = model_params['omega'].value    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model # NEED TO FIX THIS
    bm_params.u         = [u1, u2] # limb darkening coefficients # NEED TO FIX THIS

    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType)# initializes model

    return m_eclipse.light_curve(bm_params)

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, ind_kdtree, keep_inds,
                    normIntensities, xBinSize  = 0.1, yBinSize  = 0.1 ):
    intcpt = model_params['intcpt'] if 'intcpt' in model_params.keys() else 1.0 # default
    slope  = model_params['slope']  if 'slope'  in model_params.keys() else 0.0 # default
    crvtur = model_params['crvtur'] if 'crvtur' in model_params.keys() else 0.0 # default

    transit_model = transit_model_func(model_params, times[keep_inds])

    line_model    = intcpt + slope*(times[keep_inds]-times[keep_inds].mean()) \
                           + crvtur*(times[keep_inds]-times[keep_inds].mean())**2.

    # setup non-systematics model (i.e. (star + planet) / star
    model         = transit_model*line_model

    # compute the systematics model (i.e. BLISS)
    if method == 'BLISS':
        sensitivity_map = bliss.BLISS(  xcenters[keep_inds],
                                        ycenters[keep_inds],
                                        fluxes[keep_inds],
                                        knots, nearIndices,
                                        xBinSize  = xBinSize,
                                        yBinSize  = yBinSize
                                     )
    elif method == 'KRdata':
        gw_kdtree = kr.gaussian_weights_and_nearest_neighbors(xpos=xcenters[keep_inds], ypos=ycenters[keep_inds],
        npix=flux_errs[keep_inds], inds=ind_kdtree )
        sensitivity_map  = np.sum(fluxes[ind_kdtree]  * gw_kdtree, axis=1)

    elif method == 'PLD':
        PLDcoeffs = sorted([val.values  for key, val in model_params.items() if 'pld' in key.lower()])
        sensitivity_map = np.dot(normIntensities[keep_inds], PLDcoeffs)


    nSig = 10
    vbad_sm = np.where(abs(sensitivity_map - np.median(sensitivity_map)) > nSig*scale.mad(sensitivity_map))[0]
    if len(sensitivity_map)-1 in vbad_sm:
        vbad_sm = list(set(vbad_sm) - set([len(sensitivity_map)-1]))
        end_corner_case = True
    else:
        end_corner_case = False

    if 0 in vbad_sm:
        vbad_sm = list(set(vbad_sm) - set([0]))
        start_corner_case = True
    else:
        start_corner_case = False

    vbad_sm = np.array(vbad_sm)
    sensitivity_map[vbad_sm] = 0.5*(sensitivity_map[vbad_sm-1] + sensitivity_map[vbad_sm+1])

    if end_corner_case: sensitivity_map[-1] = sensitivity_map[2]
    if start_corner_case: sensitivity_map[0] = sensitivity_map[1]


    model = model * sensitivity_map

    return (model - fluxes[keep_inds]) / flux_errs[keep_inds] # should this be squared?


def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes, knots, nearIndices, ind_kdtree, keep_inds,
                                normIntensities, xBinSize  = 0.1, yBinSize  = 0.1):
    intcpt = model_params['intcpt'] if 'intcpt' in model_params.keys() else 1.0 # default
    slope  = model_params['slope']  if 'slope'  in model_params.keys() else 0.0 # default
    crvtur = model_params['crvtur'] if 'crvtur' in model_params.keys() else 0.0 # default

    transit_model = transit_model_func(model_params, times[keep_inds])

    line_model    = intcpt + slope*(times[keep_inds]-times[keep_inds].mean()) \
                           + crvtur*(times[keep_inds]-times[keep_inds].mean())**2.

    # setup non-systematics model (i.e. (star + planet) / star
    model         = transit_model*line_model

    # compute the systematics model (i.e. BLISS)
    if method == 'BLISS':
        sensitivity_map = bliss.BLISS(  xcenters[keep_inds],
                                        ycenters[keep_inds],
                                        fluxes[keep_inds],
                                        knots, nearIndices,
                                        xBinSize  = xBinSize,
                                        yBinSize  = yBinSize
                                     )
    elif method == 'KRdata':

        gw_kdtree = kr.gaussian_weights_and_nearest_neighbors(xpos=xcenters[keep_inds], ypos=ycenters[keep_inds],
        npix=flux_errs[keep_inds], inds=ind_kdtree )
        sensitivity_map  = np.sum(fluxes[ind_kdtree]  * gw_kdtree, axis=1)

    model = model * sensitivity_map

    output = {}
    output['full_model'] = model
    output['line_model'] = line_model
    output['transit_model'] = transit_model
    output['bliss_map'] = sensitivity_map

    return output

def exoparams_to_lmfit_params(planet_name):
    ep_params   = exoparams.PlanetParams(planet_name)
    iApRs       = ep_params.ar.value
    iEcc        = ep_params.ecc.value
    iInc        = ep_params.i.value
    iPeriod     = ep_params.per.value
    iTCenter    = ep_params.tt.value
    iTdepth     = ep_params.depth.value
    iOmega      = ep_params.om.value

    return iPeriod, iTCenter, iApRs, iInc, iTdepth, iEcc, iOmega

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename'   , type=str  , required=True , default='' )
ap.add_argument('-p', '--planet_name', type=str  , required=True , default='' , help='Either the string name of the planet from Exoplanets.org or a json file containing ')
ap.add_argument('-m', '--method', type=str  , required=True , default='' , help='BLISS, or KRdata')
ap.add_argument('-xb', '--xbinsize'  , type=float, required=False, default=0.1, help='Stepsize in X-sigma to space the knots')
ap.add_argument('-yb', '--ybinsize'  , type=float, required=False, default=0.1, help='Stepsize in Y-sigma to space the knots')
args = vars(ap.parse_args())

# dataDir = environ['HOME'] + "/Research/PlanetName/data/centers_and_flux_data.joblib.save"
dataDir     = args['filename']
method      = str(args['method'])
xBinSize    = float(args['xbinsize'])
yBinSize    = float(args['ybinsize'])
planet_name = args['planet_name']

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

times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds = setup_BLISS_inputs_from_file_custom(dataDir)

n_nbr   = 50
points  = np.transpose([xcenters[keep_inds], ycenters[keep_inds], flux_errs[keep_inds]])
kdtree  = spatial.cKDTree(points)
ind_kdtree  = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]

phase = ((times-init_t0)%init_period)/init_period
ph_where = np.where(phase>0.5)[0]
phase[ph_where] -= 1

normIntensities = []

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

# fluxes_std = np.std(fluxes/np.median(fluxes))

initialParams.add_many(
    ('period'   , init_period, False),
    ('tCenter'  , init_t0    , True  , init_t0 - 0.1, init_t0 + 0.1),
    ('inc'      , init_inc   , False, 80.0, 90.),
    ('aprs'     , init_aprs  , False, 0.0, 100.),
    ('tdepth'   , init_tdepth, True , 0.0, 0.3 ),
    ('edepth'   , init_fpfs  , False, 0.0, 0.05),
    ('ecc'      , init_ecc   , False, 0.0, 1.0 ),
    ('omega'    , init_omega , False, 0.0, 1.0 ),
    ('u1'       , init_u1    , True , 0.0, 1.0 ),
    ('u2'       , init_u2    , True, 0.0, 1.0 ),
    ('intcpt'   , 1.0        , True ),#, 1.0-1e-3 + 1.0+1e-3),
    ('slope'    , 0.0        , True ),
    ('crvtur'   , 0.0        , False))

n_pld = 9
for k in range(n_pld):
    initialParams.add_many(('pld{}'.format(k), True))


# Reduce the number of inputs in the objective function sent to LMFIT
#   by setting the static vectors as static in the wrapper function
partial_residuals  = partial(residuals_func,
                             times       = times,
                             xcenters    = xcenters,
                             ycenters    = ycenters,
                             fluxes      = fluxes / np.median(fluxes),
                             flux_errs   = flux_errs / np.median(fluxes),
                             knots       = knots,
                             nearIndices = nearIndices,
                             keep_inds   = keep_inds,
                             ind_kdtree  = ind_kdtree,
                             normIntensities = normIntensities)



print('Fitting the Model')
# Setup up the call to minimize the residuals (i.e. ChiSq)
mle0  = Minimizer(partial_residuals, initialParams)

start = time()

fitResult = mle0.leastsq() # Go-Go Gadget Fitting Routine

print("LMFIT operation took {} seconds".format(time()-start))

report_errors(fitResult.params)

print('Establishing the Best Fit Solution')
bf_model_set = generate_best_fit_solution(fitResult.params,
                                            times, xcenters, ycenters, fluxes / np.median(fluxes),
                                            knots, nearIndices, ind_kdtree, keep_inds, normIntensities,
                                            xBinSize  = xBinSize, yBinSize  = yBinSize)

bf_full_model = bf_model_set['full_model']
bf_line_model = bf_model_set['line_model']
bf_transit_model = bf_model_set['transit_model']
bf_bliss_map = bf_model_set['bliss_map']
print(planet_json['planet name'])
nSig = 10
good_bf = np.where(abs(bf_full_model - np.median(bf_full_model)) < nSig*scale.mad(bf_full_model))[0]



print('Plotting the Correlations')
fig1 = plt.figure()
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax21 = fig1.add_subplot(223)
ax22 = fig1.add_subplot(224)

ax11.scatter(xcenters[keep_inds][good_bf], fluxes[keep_inds][good_bf], s=0.1, alpha=0.1)
ax12.scatter(ycenters[keep_inds][good_bf], fluxes[keep_inds][good_bf], s=0.1, alpha=0.1)
ax21.scatter(xcenters[keep_inds][good_bf], ycenters[keep_inds][good_bf],
                s=0.1, alpha=0.1, c=(bf_bliss_map*bf_line_model)[good_bf])
ax22.scatter(xcenters[keep_inds][good_bf], ycenters[keep_inds][good_bf],
                s=0.1, alpha=0.1, c=(fluxes[keep_inds]-bf_full_model)[good_bf]**2)

ax11.set_title('Xcenters vs Normalized Flux')
ax21.set_title('Ycenters vs Normalized Flux')
ax12.set_title('X,Y Centers vs {} Map'.format(method))
ax22.set_title('X,Y Centers vs Residuals (Flux - {} Map)'.format(method))

nSig = 3
xCtr = xcenters[keep_inds][good_bf].mean()
xSig = xcenters[keep_inds][good_bf].std()

yCtr = ycenters[keep_inds][good_bf].mean()
ySig = ycenters[keep_inds][good_bf].std()

ax11.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
ax12.set_xlim(yCtr - nSig * ySig, yCtr + nSig * ySig)
ax21.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
ax21.set_ylim(yCtr - nSig * ySig, yCtr + nSig * ySig)
ax22.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
ax22.set_ylim(yCtr - nSig * ySig, yCtr + nSig * ySig)

mng = plt.get_current_fig_manager()
# plt.tight_layout()
plt.savefig('method{}corr'.format(method))

print('Plotting the Time Series')

fig2 = plt.figure()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)
ax1.scatter(times[keep_inds][good_bf], fluxes[keep_inds][good_bf] , s=0.1, alpha=0.1)
ax1.scatter(times[keep_inds][good_bf], bf_full_model[good_bf], s=0.1, alpha=0.1)
ax2.scatter(times[keep_inds][good_bf], (fluxes[keep_inds] - bf_full_model)[good_bf], s=0.1, alpha=0.1)

ax1.set_title('{} Raw CH2 Light Curve with {} + Linear + BATMAN Model'.format(planet_name, method))
ax2.set_title('{} Raw CH2 Residuals (blue - orange above)'.format(planet_name))
plt.savefig('methodKRDATAtimes')

mng = plt.get_current_fig_manager()
# plt.tight_layout()
mle0.params.add('f', value=1, min=0.001, max=2)

def logprior_func(p):
    return 0

def lnprob(p):
    logprior = logprior_func(p)
    if not np.isfinite(logprior):
        return -np.inf

    resid = partial_residuals(p)
    s = p['f']
    resid *= 1 / s
    resid *= resid
    resid += np.log(2 * np.pi * s**2)
    return -0.5 * np.sum(resid) + logprior


mini  = Minimizer(lnprob, mle0.params)

start = time()

#import emcee
#res = emcee.sampler(lnlikelihood = lnprob, lnprior=logprior_func)

res   = mini.emcee(params=mle0.params, steps=100, nwalkers=100, burn=1, thin=10, ntemps=1,
                    pos=None, reuse_sampler=False, workers=1, float_behavior='posterior',
                    is_weighted=True, seed=None)

#
print("MCMC operation took {} seconds".format(time()-start))

joblib.dump(res,'emceefinal.joblib.save')
# corner_use    = [1, 4,5,]
res_var_names = np.array(res.var_names)
res_flatchain = np.array(res.flatchain)
res_df = DataFrame(res_flatchain, columns=res_var_names)
res_df = res_df.drop(['u2','slope'], axis=1)
print(res_df)
# res_flatchain.T[corner_use].shape
corner_kw = dict(levels=[0.68, 0.95, 0.997], plot_datapoints=False, smooth=True, bins=30)

corner.corner(res_df, color='darkblue', **corner_kw, range=[(54945,54990),(0.01357,0.01385),(0.1097,0.11035),(0.996,1.002), (0.998,1.003)], plot_density=False, fill_contours=True)
plt.savefig('method{}corner'.format(method))
