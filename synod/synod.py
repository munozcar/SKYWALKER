'''
SYNOD: Spitzer Systematcs Noise Decorrelation. Written by Carlos E. Munoz-Romero and Jonathan D. Fraine, 2018.
Usage: python synod.py -f photometry_file -p planet_params_file -m method -xb x_bin_size -yb y_bin_size
'''

# Import internal dependencies and methods.
from . import bliss
from . import krdata as kr
from . import utils
from . import models

# Import external dependencies and methods.
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
from scipy.interpolate import CubicSpline
from sklearn.externals import joblib
from functools import partial
from lmfit import Parameters, Minimizer, report_errors
from scipy import spatial
from statsmodels.robust import scale
from time import time

# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0
# Function definitions

def add_cubicspline_to_phase_curve_model(model_params, times, init_t0, phase_curve_model, eclipse_model):
    
    ecl_bottom = eclipse_model == eclipse_model.min()
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
    
    y1_0 = phase_curve_model[t1_0]
    y2_0 = 1 #- model_params['edepth'].value
    y3_0 = 1 #- model_params['edepth'].value
    y4_0 = phase_curve_model[t4_0]
    
    y1_1 = phase_curve_model[t1_1]
    y2_1 = 1 #- model_params['edepth'].value
    y3_1 = 1 #- model_params['edepth'].value
    y4_1 = phase_curve_model[t4_1]
    
    x1_0 = times[t1_0]
    x2_0 = times[t2_0]
    x3_0 = times[t3_0]
    x4_0 = times[t4_0]
    
    x1_1 = times[t1_1]
    x2_1 = times[t2_1]
    x3_1 = times[t3_1]
    x4_1 = times[t4_1]
    
    ecl_top = np.where(eclipse_model == eclipse_model.max())[0]
    ecl_bottom = np.where(ecl_bottom)[0]
    
    cs_idx = np.sort(np.hstack([ecl_top, ecl_bottom]))
    
    output_model = phase_curve_model.copy()
    
    output_model[t2_0:t3_0] = y2_0 # == 1.0
    output_model[t2_1:t3_1] = y2_1 # == 1.0
    
    cs_local = CubicSpline(times[cs_idx], output_model[cs_idx])
    print("THIS IS BROKEN")
    return cs_local(times, 1)

def add_trap_to_phase_curve_model(model_params, times, init_t0, phase_curve_model, eclipse_model):
    
    ecl_bottom = eclipse_model == eclipse_model.min()
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
    
    y1_0 = phase_curve_model[t1_0]
    y2_0 = 1 #- model_params['edepth'].value
    y3_0 = 1 #- model_params['edepth'].value
    y4_0 = phase_curve_model[t4_0]
    
    y1_1 = phase_curve_model[t1_1]
    y2_1 = 1 #- model_params['edepth'].value
    y3_1 = 1 #- model_params['edepth'].value
    y4_1 = phase_curve_model[t4_1]
    
    x1_0 = times[t1_0]
    x2_0 = times[t2_0]
    x3_0 = times[t3_0]
    x4_0 = times[t4_0]
    
    x1_1 = times[t1_1]
    x2_1 = times[t2_1]
    x3_1 = times[t3_1]
    x4_1 = times[t4_1]    
    
    ingress_slope_0 = (y2_0 - y1_0) / (x2_0 - x1_0)
    ingress_slope_1 = (y2_1 - y1_1) / (x2_1 - x1_1)
    
    egress_slope_0 = (y4_0 - y3_0) / (x4_0 - x3_0)
    egress_slope_1 = (y4_1 - y3_1) / (x4_1 - x3_1)
    
    output_model = phase_curve_model.copy()
    
    output_model[t2_0:t3_0] = y2_0 # == 1.0
    output_model[t2_1:t3_1] = y2_1 # == 1.0
    
    output_model[t1_0:t2_0] = ingress_slope_0 * (times[t1_0:t2_0]-x2_0) + y2_0
    output_model[t1_1:t2_1] = ingress_slope_1 * (times[t1_1:t2_1]-x2_1) + y2_1
    output_model[t3_0:t4_0] = egress_slope_0 * (times[t3_0:t4_0]-x3_0) + y3_0
    output_model[t3_1:t4_1] = egress_slope_1 * (times[t3_1:t4_1]-x3_1) + y3_1
    
    return output_model

def compute_full_model(model_params, times, include_transit = True, 
                        include_eclipse = True, include_phase_curve = True, 
                        include_polynomial = True, eclipse_option = 'trapezoid',
                        subtract_edepth = True, return_case = None,
                        use_trap = False):
    
    init_t0 = model_params['tCenter']
    
    if 'tdepth' not in model_params.keys(): include_transit = False
    if 'edepth' not in model_params.keys(): include_eclipse = False
    if 'intercept' not in model_params.keys(): include_polynomial = False
    if 'cosAmp' not in model_params.keys(): include_phase_curve = False
    
    line_model = models.line_model_func(model_params, times) if include_polynomial else 1.0
    
    transit_model = models.transit_model_func(model_params, times, init_t0, transitType='primary')  if include_transit else 1.0
    
    if include_eclipse:
        if use_trap:
            eclipse_model = models.trapezoid_model(model_params, times, init_t0)  #if include_eclipse else 1.0
        else:
            eclipse_model = models.transit_model_func(model_params, times, init_t0, transitType='secondary')  #if include_eclipse else 1.0
    else:
        eclipse_model = 1.0
    
    phase_curve_model = models.phase_curve_func(model_params, times, init_t0) if include_phase_curve else 1.0
    
    if subtract_edepth: eclipse_model = eclipse_model - model_params['edepth'].value
    
    where_eclipse = np.where(eclipse_model < eclipse_model.max())[0]
    
    ecl_bottom = eclipse_model == eclipse_model.min()
    # model_params['edepth'].value = phase_curve_model[ecl_bottom].mean() - 1.0
    model_params['edepth'].value = phase_curve_model[ecl_bottom].max() - 1.0 # ??
    
    mutl_ecl = True
    try:
        if model_params['edepth'].value > 0.0 and np.isfinite(model_params['edepth'].value):
            mutl_ecl = False
            if eclipse_option == 'cubicspline':
                phase_curve_model = add_cubicspline_to_phase_curve_model(model_params, times, init_t0, phase_curve_model, eclipse_model)
            elif eclipse_option == 'trapezoid':
                phase_curve_model = add_trap_to_phase_curve_model(model_params, times, init_t0, phase_curve_model, eclipse_model)
        else:
            print('Edepth: {}'.format(model_params['edepth'].value))
            mutl_ecl = False
    except Exception as e:
        mutl_ecl = False
        
        print('\n[WARNING] Failure Occured with {}'.format(str(e)))
        print('[WARNING] Model Params at Failure were')
        for val in model_params.values(): 
            print('\t\t{:11}: {:3}\t[{:5}, {:5}]\t{}'.format(val.name, str(val.value)[:10], str(val.min)[:5], str(val.max)[:5], str(val.vary)))
        
        print('\n[WARNING] BATMAN Eclipse Model Mean: {}'.format(eclipse_model.mean()))
    
    if np.allclose(phase_curve_model, np.ones(phase_curve_model.size)): mutl_ecl = True
    
    # non-systematics model (i.e. (star + planet) / star
    physical_model = transit_model*line_model*phase_curve_model
    
    if mutl_ecl: physical_model = physical_model*eclipse_model
    
    if return_case == 'dict':
        output = {}
        # output['line_model'] = line_model
        output['physical_model'] = physical_model
        output['transit_model'] = transit_model
        output['eclipse_model'] = eclipse_model
        output['line_model'] = line_model
        output['phase_curve_model'] = phase_curve_model
        
        return output
    else:
        return physical_model

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, keep_inds, 
                    method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, pld_intensities=None, 
                    x_bin_size  = 0.1, y_bin_size  = 0.1, transit_indices=None, include_transit = True, 
                    include_eclipse = True, include_phase_curve = True, include_polynomial = True, 
                    testing_model = False, eclipse_option = 'trapezoid', use_trap = False):
    
    physical_model = compute_full_model(model_params, times, include_transit = include_transit, 
                        include_eclipse = include_eclipse, include_phase_curve = include_phase_curve, 
                        include_polynomial = include_polynomial, eclipse_option = eclipse_option)
    
    if testing_model: return physical_model
    
    # compute the systematics model
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "Invalid method."
    residuals = fluxes / physical_model
    
    sensitivity_map = models.compute_sensitivity_map(model_params=model_params, 
                                                     method=method, 
                                                     xcenters=xcenters, 
                                                     ycenters=ycenters, 
                                                     residuals=residuals, 
                                                     knots=knots, 
                                                     nearIndices=nearIndices, 
                                                     xBinSize=x_bin_size, 
                                                     yBinSize=y_bin_size, 
                                                     ind_kdtree=ind_kdtree, 
                                                     gw_kdtree=gw_kdtree, 
                                                     pld_intensities=pld_intensities, 
                                                     model=physical_model)
    
    if 't_start' in model_params.keys() and 'weirdslope' in model_params.keys() and 'weirdintercept' in model_params.keys():
        weirdness = np.ones(times.size)
        weirdness[times > model_params['t_start']] = model_params['weirdslope']*(times[times > model_params['t_start']]-times.mean()) + model_params['weirdintercept']
    else:
        weirdness = 1.0
    
    model = physical_model*sensitivity_map*weirdness
    
    return (model - fluxes) / flux_errs 

def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes, knots, keep_inds, 
                                method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, 
                                pld_intensities=None, x_bin_size  = 0.1, y_bin_size  = 0.1, 
                                transit_indices=None, include_transit = True, include_eclipse = True, 
                                include_phase_curve = True, include_polynomial = True, eclipse_option = 'trapezoid'):
    
    output = compute_full_model(model_params, times, include_transit = include_transit, include_eclipse = include_eclipse, 
                                include_phase_curve = include_phase_curve, include_polynomial = include_polynomial, 
                                eclipse_option = eclipse_option, return_case='dict')
    
    # compute the systematics model
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "Invalid method."
    
    residuals = fluxes / output['physical_model']
    sensitivity_map = models.compute_sensitivity_map(model_params=model_params, method=method, xcenters=xcenters, ycenters=ycenters, 
                                                        residuals=residuals, knots=knots, nearIndices=nearIndices, xBinSize=x_bin_size, 
                                                        yBinSize=y_bin_size, ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree, 
                                                        pld_intensities=pld_intensities, model=output['physical_model'])
    
    model = output['physical_model']*sensitivity_map
    
    output['full_model'] = model
    output['sensitivity_map'] = sensitivity_map
    
    return output
