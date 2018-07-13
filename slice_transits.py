import argparse
import BLISS as bliss
import exoparams
import json
import numpy as np
from sklearn.externals import joblib
from scipy import spatial
from statsmodels.robust import scale

y, x = 0, 1
ppm = 1e6


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
        knots (nDarray): locations and initial flux values (weights) for interpolation grid
        nearIndices (nDarray): nearest neighbour indices per point for location of nearest knots
        keep_inds (list): list of indicies to keep within the thresholds set
    """
    times, xcenters, ycenters, fluxes, flux_errs = bliss.extractData(dataDir)

    keep_inds = bliss.removeOutliers(xcenters, ycenters, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange)
    # , fSigmaRange)

    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)

    return times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds


def exoparams_to_lmfit_params(planet_name):
    ep_params = exoparams.PlanetParams(planet_name)
    iApRs = ep_params.ar.value
    iEcc = ep_params.ecc.value
    iInc = ep_params.i.value
    iPeriod = ep_params.per.value
    iTCenter = ep_params.tt.value
    iTdepth = ep_params.depth.value
    iOmega = ep_params.om.value

    return iPeriod, iTCenter, iApRs, iInc, iTdepth, iEcc, iOmega


try:
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filename', type=str, required=True, default='',
                    help='File storing the times, xcenters, ycenters, fluxes, flux_errs')
    ap.add_argument('-p', '--planet_name', type=str, required=True, default='',
                    help='Either the string name of the planet from Exoplanets.org or a json file containing ')
    args = vars(ap.parse_args())

    dataDir = args['filename']
    planet_name = args['planet_name']
except:
    dataDir = '../../data/group{}_gsc.joblib.save'.format(4)
    planet_name = 'gj1214b_planet_params.json'

init_u1, init_u2, init_u3, init_u4, init_fpfs = None, None, None, None, None

if planet_name[-5:] == '.json':
    with open(planet_name, 'r') as file_in:
        planet_json = json.load(file_in)
    init_period = planet_json['period']
    init_t0 = planet_json['t0']
    init_aprs = planet_json['aprs']
    init_inc = planet_json['inc']

    if 'tdepth' in planet_json.keys():
        init_tdepth = planet_json['tdepth']
    elif 'rprs' in planet_json.keys():
        init_tdepth = planet_json['rprs'] ** 2
    elif 'rp' in planet_json.keys():
        init_tdepth = planet_json['rp'] ** 2
    else:
        raise ValueError("Eitehr `tdepth` or `rprs` or `rp` (in relative units) \
                            must be included in {}".format(planet_name))

    init_fpfs = planet_json['fpfs'] if 'fpfs' in planet_json.keys() else 500 / ppm
    init_ecc = planet_json['ecc']
    init_omega = planet_json['omega']
    init_u1 = planet_json['u1'] if 'u1' in planet_json.keys() else None
    init_u2 = planet_json['u2'] if 'u2' in planet_json.keys() else None
    init_u3 = planet_json['u3'] if 'u3' in planet_json.keys() else None
    init_u4 = planet_json['u4'] if 'u4' in planet_json.keys() else None

    if 'planet name' in planet_json.keys():
        planet_name = planet_json['planet name']
    else:
        # Assume the json file name is the planet name
        #   This is a bad assumption; but it is one that users will understand
        print("'planet name' is not inlcude in {};".format(planet_name), end=" ")
        planet_name = planet_name.split('.json')[0]
        print(" assuming the 'planet name' is {}".format(planet_name))
else:
    init_period, init_t0, init_aprs, init_inc, init_tdepth, init_ecc, init_omega = exoparams_to_lmfit_params(
        planet_name)

init_fpfs = 500 / ppm if init_fpfs is None else init_fpfs
init_u1 = 0.1 if init_u1 is None else init_u1
init_u2 = 0.0 if init_u2 is None else init_u2
init_u3 = 0.0 if init_u3 is None else init_u3
init_u4 = 0.0 if init_u4 is None else init_u4

print('Acquiring Data')
times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds = setup_BLISS_inputs_from_file(dataDir)

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

phase = ((times - init_t0) % init_period) / init_period
phase[phase > 0.5] -= 1.0

transit_phase = 0.1
ph_transits = np.where(abs(phase) < transit_phase)[0]

ph_diff_times = np.diff(times[ph_transits] * 86400)
med_ph_diff_times = np.median(ph_diff_times)
std_ph_diff_times = np.std(ph_diff_times)

nSig = 10
ph_where_transits = np.where(abs(ph_diff_times) > nSig * std_ph_diff_times)[0]

if len(ph_where_transits) == len(ph_transits) - 1 or ph_where_transits == []:
    print('There is probably only 1 transit in this data set')
    print('\tWe will store *only* the phase range equivalent to that single transit')
    ph_where_transits = [len(ph_transits) - 1]
    single_transit = True
else:
    single_transit = False
ntransits = len(ph_where_transits)

idx_start = ph_transits[0]
for kt in range(ntransits):
    idx_end = ph_transits[ph_where_transits[kt]]
    current_transit = {}
    current_transit[b'times'] = times[idx_start:idx_end]
    current_transit[b'xcenters'] = xcenters[idx_start:idx_end]
    current_transit[b'ycenters'] = ycenters[idx_start:idx_end]
    current_transit[b'flux'] = fluxes[idx_start:idx_end]

    curr_name = dataDir.replace('_gsc.joblib.save', '_gsc_transit{}.joblib.save'.format(kt))

    print("Saving transit{} to {}".format(kt, curr_name))
    joblib.dump(current_transit, curr_name)
    if not single_transit and idx_end != len(ph_transits) - 1:
        idx_start = ph_transits[ph_where_transits[kt] + 1]
    else:
        # CORNER CASE
        error_messages = {True: "There is probably only one transits in this data",
                          False: "The transit probably meets the end of the data"}

        print(error_messages[single_transit])

        ph_transits[-1]

if not single_transit and idx_start != len(ph_transits) - 1:
    '''Catch the last transit'''
    kt = kt + 1  #
    idx_end = ph_transits[-1]
    current_transit = {}
    current_transit[b'times'] = times[idx_start:idx_end]
    current_transit[b'xcenters'] = xcenters[idx_start:idx_end]
    current_transit[b'ycenters'] = ycenters[idx_start:idx_end]
    current_transit[b'flux'] = fluxes[idx_start:idx_end]

    curr_name = dataDir.replace('_gsc.joblib.save', '_gsc_transit{}.joblib.save'.format(kt))
    print("Saving transit{} to {}".format(kt, curr_name))
joblib.dump(current_transit, curr_name)
