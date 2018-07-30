import batman
import numpy as np
import bliss
import utils
import krdata as kr
from statsmodels.robust import scale
import matplotlib.pyplot as plt
import numpy.linalg as linear
# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0

def transit_model_func(model_params, times, init_t0=0.0, ldtype='quadratic', transitType='primary'):
    """
        Args:
            model_params: Parameters() object with orbital properties for a given exoplanet.
            times: array of dates in units of days utilized for the photometry time series.
            init_t0: transit center time.
            ldtype: transit model type.
            transitType: 'primary' for transit, 'secondary' for eclipse.
        Returns:
            The Dark Knight phase curve model.
    """
    # Transit Parameters
    u1 = model_params['u1'].value
    u2 = model_params['u2'].value

    bm_params = batman.TransitParams() # object to store transit parameters

    if 'edepth' in model_params.keys() and model_params['edepth'] > 0:
        bm_params.t_secondary = model_params['deltaTc'] + init_t0 + 0.5*model_params['period'] + model_params['deltaEc']
        # model_params['period']*delta_phase
    else:
        model_params.add('edepth', 0.0, False)

    rprs = np.sqrt(model_params['tdepth'].value)
    bm_params.per = model_params['period'].value            # orbital period
    bm_params.t0 = model_params['deltaTc'].value + init_t0  # time of inferior conjunction
    bm_params.inc = model_params['inc'].value               # inclunaition in degrees
    bm_params.a = model_params['aprs'].value                # semi-major axis (in units of stellar radii)
    bm_params.rp = rprs                                     # planet radius (in units of stellar radii)
    bm_params.fp = model_params['edepth'].value             # planet radius (in units of stellar radii)
    bm_params.ecc = model_params['ecc'].value               # eccentricity
    bm_params.w = model_params['omega'].value               # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype                            # limb darkening model # NEED TO FIX THIS
    bm_params.u = [u1, u2]                                  # limb darkening coefficients # NEED TO FIX THIS
    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType) # initializes model

    return m_eclipse.light_curve(bm_params)


def line_model_func(model_params, ntransits, transit_indices, times, token=False):
    intercepts = []
    coeffs_line_list = []

    for k in range(ntransits):
        intercepts.append(model_params['intcept{}'.format(k)] if 'intcept{}'.format(k) in model_params.keys() else 1.0 )
        slope = model_params['slope{}'.format(k)] if 'slope{}'.format(k) in model_params.keys() else 0.0
        crvtur = model_params['crvtur{}'.format(k)] if 'crvtur{}'.format(k) in model_params.keys() else 0.0
        coeffs_line_list.append([slope, crvtur])


    total_line = []
    for ki, intcpt in enumerate(intercepts):
        times_now = times[transit_indices[ki][0]:transit_indices[ki][1]]
        # Flat line
        line_model = np.array([intcpt for x in np.zeros(len(times_now))])
        # slope * [times-shift] + curvatue * [times-shift]**2
        for kc,c_now in enumerate(coeffs_line_list[ki]):
            if c_now != zero:
                slant = float(c_now)*(times_now-times_now.mean())**(kc+1)

            else:
                slant = 0*times_now
        line_model = line_model + slant

        total_line = total_line + list(line_model)

    return np.array(total_line)

def compute_sensitivity_map(model_params, method, xcenters, ycenters, residuals, knots, nearIndices, xBinSize, yBinSize, ind_kdtree, gw_kdtree, pld_intensities):
    if method == 'bliss' :
        normFactor = (1/xBinSize) * (1/yBinSize)
        sensitivity_map = bliss.BLISS(xcenters, ycenters, residuals, knots, nearIndices, xBinSize=xBinSize, yBinSize=xBinSize, normFactor=normFactor)
    elif method == 'krdata':
        sensitivity_map  = np.sum(residuals[ind_kdtree]  * gw_kdtree, axis=1)
    elif method == 'pld':
        PLDcoeffs = sorted([val for key, val in model_params.items() if 'pld' in key.lower()])
        sensitivity_map = np.dot(PLDcoeffs, pld_intensities)
    else:
        print('INVALID METHOD: ABORT')
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

    return sensitivity_map

def add_line_params(model_params, phase, times, transitType='primary'):

    transit_phase = 0.1
    if transitType == 'primary':
        ph_transits = np.where(abs(phase) < transit_phase)[0]
    elif transitType == 'secondary':
        ph_transits = np.where(abs(phase-0.5) < transit_phase)[0]

    day_to_seconds = 86400
    ph_diff_times = np.diff(times[ph_transits] * day_to_seconds)
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
    print('Found {} transits'.format(ntransits))
    transit_indices = []
    idx_start = ph_transits[0]
    for kt in range(ntransits):
        idx_end = ph_transits[ph_where_transits[kt]]
        transit_indices.append([idx_start,idx_end+1])
        model_params.add_many(('intcept{}'.format(kt), 1.0, True))
        model_params.add_many(('slope{}'.format(kt), 0.0, True))
        model_params.add_many(('crvtur{}'.format(kt), 0.0, False))
        if not single_transit and idx_end != len(ph_transits) - 1:
            idx_start = ph_transits[ph_where_transits[kt] + 1]
        else:
            # CORNER CASE
            error_messages = {True: "There is probably only one eclipses in this data",
                              False: "The eclipse probably meets the end of the data"}

            print(error_messages[single_transit])

            ph_transits[-1]

    if not single_transit and idx_start != len(ph_transits) - 1:
        '''Catch the last ocurrence'''
        kt = kt + 1  #
        model_params.add_many(('intcept{}'.format(kt), 1.0, True))
        model_params.add_many(('slope{}'.format(kt), 0.0, True))
        model_params.add_many(('crvtur{}'.format(kt), 0.0, False))
        idx_end = ph_transits[-1]
        transit_indices.append([idx_start,idx_end+1])

    return model_params, transit_indices

def add_pld_params(model_params):
    n_pld = 9
    for k in range(n_pld):
        model_params.add_many(('pld{}'.format(k), 1.1, True))
    return model_params
