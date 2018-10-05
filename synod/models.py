import batman
import numpy as np
from . import bliss
from . import utils
from . import krdata as kr

from functools import partial
from statsmodels.robust import scale
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

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
    
    if 'deltaTc' in model_params.keys():
        if 'deltaEc' in model_params.keys():
            bm_params.t_secondary = model_params['deltaTc'] + init_t0 + 0.5*model_params['period'] + model_params['deltaEc']
        else:
            bm_params.t_secondary = model_params['deltaTc'] + init_t0 + 0.5*model_params['period']
    else:
        bm_params.t_secondary = init_t0 + 0.5*model_params['period']
    
    if 'edepth' not in model_params.keys(): model_params.add('edepth', 0.0, False)
    
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
    
    return m_eclipse.light_curve(bm_params)# + oot_offset

eclipse_model_func = partial(transit_model_func, transitType='secondary')

def line_model_func_multi(model_params, ntransits, transit_indices, times):
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
        times_now = times_now - times_now.mean()
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

def line_model_func(model_params, times):
    intercept = model_params['intercept'] if 'intercept' in model_params.keys() else 1.0
    slope = model_params['slope'] if 'slope' in model_params.keys() else 0.0
    curvature = model_params['curvature'] if 'curvature' in model_params.keys() else 0.0
    
    line_model = intercept
    line_model = line_model + slope*(times-times.mean()) if slope is not 0.0 else line_model
    line_model = line_model + curvature*(times-times.mean())**2 if curvature is not 0.0 else line_model
    
    return line_model

def phase_curve_func(model_params, times, init_t0):
    
    if 'period' not in model_params.keys(): raise Exception('`period` not included in `model_params`')
    
    if 'deltaTc' in model_params.keys() and 'deltaEc' in model_params.keys():
        t_secondary = init_t0 + model_params['deltaTc'] + 0.5*model_params['period'] + model_params['deltaEc']
    else:
        t_secondary = init_t0 + 0.5*model_params['period']
    
    ang_freq = 2*np.pi / model_params['period']
    if 'cosPhase' in model_params.keys() and 'cosAmp' in model_params.keys():
        half = 0.5 # necessary because the "amplitude" of a cosine is HALF the "amplitude"" of the phase curve
        # phase_curve = half*model_params['cosAmp']*np.cos(ang_freq * (times - t_secondary) + model_params['cosPhase'])
        phase_curve = half*model_params['cosAmp']*np.cos(ang_freq * (times - t_secondary + model_params['cosPhase']))
    elif 'sinAmp' in model_params.keys() and 'cosAmp' in model_params.keys():
        phase_curve = model_params['cosAmp']*np.cos(ang_freq * (times - t_secondary)) + model_params['sinAmp']*np.sin(ang_freq * (times - t_secondary))
    elif 'sinAmp1' in model_params.keys() and 'cosAmp1' in model_params.keys():
        if 'sinAmp2' in model_params.keys() and 'cosAmp2' in model_params.keys():
            phase_curve = model_params['cosAmp1']*np.cos(ang_freq * (times - t_secondary)) + model_params['sinAmp1']*np.sin(ang_freq * (times - t_secondary)) + \
                   model_params['cosAmp2']*np.cos(2*ang_freq * (times - t_secondary)) + model_params['sinAmp2']*np.sin(2*ang_freq * (times - t_secondary))
        else:
            phase_curve = model_params['cosAmp1']*np.cos(ang_freq * (times - t_secondary)) + model_params['sinAmp1']*np.sin(ang_freq * (times - t_secondary))
    else:
        phase_curve = np.array(0)
    
    # The `+ 1.0 - phase_curve.min()` term is required because the "phase curve" minimizes to 1.0;
    #   but the cosine function minimizes at -cos_amplitude / 2
    # 
    # This form ensures that the minimum phase curve will always be exactly 1.0
    return phase_curve + 1.0 - phase_curve.min() + abs(model_params['night_flux'].value)

def inc2b(inc, aRs, e = 0, w = 0):
    #convert_inc_to_b
    if e == 0:
        return aRs * cos(inc)
    elif w == 0:
        return aRs * cos(inc) * (1 - e*e)
    else:
        return aRs * cos(inc) * (1 - e*e) / (1.0 - e*sin(w))

def transit_duration(period, aprs, rprs, inc):
    ''' Compute the transit duration from tangent (t1) to tangent (t4)
    '''
    
    # circular orbits only for now
    b_imp = inc2b(inc, aprs)
    
    out_sin = period/np.pi
    in_sin = np.sqrt((1+rprs)**2 - b_imp) / aprs / np.sin(inc)
    
    return out_sin * np.arcsin(in_sin)

def transit_full(period, aprs, rprs, inc):
    ''' Compute the transit duration at full coverage, from t2 to t3
    '''
    
    # circular orbits only for now
    b_imp = inc2b(inc, aprs)
    
    out_sin = period/np.pi
    in_sin = np.sqrt((1-rprs)**2 - b_imp) / aprs / np.sin(inc)
    
    return out_sin * np.arcsin(in_sin)

def trapezoid_model(model_params, times, init_t0, delta_eclipse_time=0.0, eclipse_model=None):
    
    if eclipse_model is None: 
        del delta_eclipse_time
        delta_eclipse_time = 0.0 # reset to avoid later double dipping on the shift in eclipse location
        eclipse_model = eclipse_model_func(model_params, times, init_t0)
    
    ecl_bottom = eclipse_model < eclipse_model.min() + ftol
    in_eclipse = eclipse_model < eclipse_model.max()
    
    eclipse0 = in_eclipse * (times < times.mean())
    eclipse1 = in_eclipse * (times > times.mean())
    
    t1_0 = np.where(eclipse0)[0][0]
    t2_0 = np.where(eclipse0*ecl_bottom)[0][0]
    t3_0 = np.where(eclipse0*ecl_bottom)[0][-1]
    t4_0 = np.where(eclipse0)[0][-1]
    
    t1_1 = np.where(eclipse1)[0][0]
    t2_1 = np.where(eclipse1*ecl_bottom)[0][0]
    t3_1 = np.where(eclipse1*ecl_bottom)[0][-1]
    t4_1 = np.where(eclipse1)[0][-1]
    
    trap_model = np.ones(times.size) + model_params['edepth'].value
    
    y1 = 1.0 + model_params['edepth'].value
    y2 = 1 #- model_params['edepth'].value
    y3 = 1 #- model_params['edepth'].value
    y4 = 1.0 + model_params['edepth'].value
    
    x1_0 = times[t1_0] + delta_eclipse_time
    x2_0 = times[t2_0] + delta_eclipse_time
    x1_1 = times[t1_1] + delta_eclipse_time
    x2_1 = times[t2_1] + delta_eclipse_time
    x3_0 = times[t3_0] + delta_eclipse_time
    x4_0 = times[t4_0] + delta_eclipse_time
    x3_1 = times[t3_1] + delta_eclipse_time
    x4_1 = times[t4_1] + delta_eclipse_time
    
    ingress_slope_0 = (y2 - y1) / (x2_0 - x1_0)
    ingress_slope_1 = (y2 - y1) / (x2_1 - x1_1)
    
    egress_slope_0 = (y3 - y4) / (x3_0 - x4_0)
    egress_slope_1 = (y3 - y4) / (x3_1 - x4_1)
    
    trap_model[t2_0:t3_0] = y2
    trap_model[t2_1:t3_1] = y2
    
    trap_model[t1_0:t2_0] = ingress_slope_0 * (times[t1_0:t2_0]-x2_0) + y2
    trap_model[t1_1:t2_1] = ingress_slope_1 * (times[t1_1:t2_1]-x2_1) + y2
    trap_model[t3_0:t4_0] = egress_slope_0 * (times[t3_0:t4_0]-x3_0) + y3
    trap_model[t3_1:t4_1] = egress_slope_1 * (times[t3_1:t4_1]-x3_1) + y3
    
    return trap_model

def compute_sensitivity_map(model_params, method, xcenters, ycenters, residuals, knots, nearIndices, xBinSize, yBinSize, ind_kdtree, gw_kdtree, pld_intensities, model):
    if 'bliss' in method.lower():
        normFactor = (1/xBinSize) * (1/yBinSize)
        sensitivity_map = bliss.BLISS(xcenters, ycenters, residuals, knots, nearIndices, xBinSize=xBinSize, yBinSize=xBinSize, normFactor=normFactor)
    elif 'krdata' in method.lower():
        sensitivity_map  = np.sum(residuals[ind_kdtree]  * gw_kdtree, axis=1)
    elif 'pld' in method.lower():
        PLDcoeffs = [val.value for val in model_params.values() if 'pld' in val.name.lower()]
        sensitivity_map = np.dot(PLDcoeffs, pld_intensities)
    else:
        print('INVALID METHOD: ABORT!')
    
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

def add_pld_params(model_params, fluxes, pld_intensities, 
                    n_pld = 9, order=3, add_unity = True, 
                    do_pca=True, do_ica=False, do_std=True, 
                    pca_cut=False, n_ppm = 1.0, start_unity=False, 
                    verbose=False):
    
    # Make a local copy
    pld_intensities = pld_intensities.copy()
    
    if len(pld_intensities) != n_pld * order:
        pld_intensities = np.vstack([list(pld_intensities**k) for k in range(1,order+1)])
    
    # check that the second set is the square of the first set, and so onself.
    for k in range(order): assert(np.allclose(pld_intensities[:n_pld]**(k+1), pld_intensities[k*n_pld:(k+1)*n_pld]))
    
    if do_pca or do_ica: do_std = True
    
    stdscaler = StandardScaler()
    pld_intensities = stdscaler.fit_transform(pld_intensities.T) if do_std else pld_intensities.T
    
    if do_pca:
        pca = PCA()
        
        pld_intensities = pca.fit_transform(pld_intensities)
        
        evrc = pca.explained_variance_ratio_.cumsum()
        n_pca = np.where(evrc > 1.0-n_ppm/ppm)[0].min()
        if pca_cut: pld_intensities = pld_intensities[:,:n_pca]
        
        if verbose: print(evrc, n_pca)
    
    if do_ica:
        ica = FastICA()
        
        pld_intensities = ica.fit_transform(pld_intensities)
        
        # evrc = ica.explained_variance_ratio_.cumsum()
        # n_ica = np.where(evrc > 1.0-n_ppm/ppm)[0].min()
        # if ica_cut: pld_intensities = pld_intensities[:,:n_ica]
        #
        # if verbose: print(evrc, n_ica)
    
    if add_unity: pld_intensities = np.vstack([pld_intensities.T, np.ones(pld_intensities.shape[0])]).T
    
    pld_coeffs = np.linalg.lstsq(pld_intensities, fluxes)[0] if not start_unity else np.ones(pld_intensities.shape[1]) / pld_intensities.shape[1.0]
    
    n_pld_out = n_pca if do_pca and pca_cut else n_pld*order
    
    for k in range(n_pld_out): model_params.add_many(('pld{}'.format(k), pld_coeffs[k], True))
    
    # if add_unity: model_params.add_many(('pld{}'.format(n_pld_out), pld_coeffs[n_pld_out], True)) # FINDME: Maybe make min,max = 0,2 or = 0.9,1.1
    if add_unity: model_params.add_many(('pldBase', pld_coeffs[n_pld_out], True)) # FINDME: Maybe make min,max = 0,2 or = 0.9,1.1
    
    if verbose: [print('{:5}: {}'.format(val.name, val.value)) for val in model_params.values() if 'pld' in val.name.lower()];
    
    return model_params, pld_intensities.T