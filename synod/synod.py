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

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, keep_inds, 
                    method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, pld_intensities=None, 
                    x_bin_size  = 0.1, y_bin_size  = 0.1, transit_indices=None, include_transit = True, 
                    include_eclipse = True, include_phase_curve = True, include_polynomial = True):
    
    init_t0 = model_params['tCenter']
    ntransits = len(transit_indices)
    
    if 'tdepth' not in model_params.keys(): include_transit = False
    if 'edepth' not in model_params.keys(): include_eclipse = False
    if 'intercept' not in model_params.keys(): include_polynomial = False
    if 'cosAmp' not in model_params.keys(): include_phase_curve = False
    
    line_model = models.line_model_func(model_params, ntransits, transit_indices, times) if include_polynomial else 1.0
    
    transit_model = models.transit_model_func(model_params, times, init_t0, transitType='primary') if include_transit else 1.0
    
    eclipse_model = models.transit_model_func(model_params, times, init_t0, transitType='secondary') if include_eclipse else 1.0
    
    phase_curve_model = models.phase_curve_func(model_params, times, init_t0) if include_phase_curve else 1.0
    
    # non-systematics model (i.e. (star + planet) / star
    physical_model = transit_model*eclipse_model*line_model*phase_curve_model
    
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
    
    model = physical_model*sensitivity_map
    
    return (model - fluxes) / flux_errs 

def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes, knots, keep_inds, 
                                method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, 
                                pld_intensities=None, x_bin_size  = 0.1, y_bin_size  = 0.1, 
                                transit_indices=None):
    
    ntransits = len(transit_indices)
    init_t0 = model_params['tCenter']
    
    if ntransits > 1: print('SYNOD has fitted {} transits or eclipses'.format(ntransits))
    
    if 'tdepth' not in model_params.keys(): include_transit = False
    if 'edepth' not in model_params.keys(): include_eclipse = False
    if 'intercept' not in model_params.keys(): include_polynomial = False
    if 'cosAmp' not in model_params.keys(): include_phase_curve = False
    
    line_model = models.line_model_func(model_params, ntransits, transit_indices, times) if include_polynomial else 1.0
    
    transit_model = models.transit_model_func(model_params, times, init_t0, transitType='primary') if include_transit else 1.0
    
    eclipse_model = models.transit_model_func(model_params, times, init_t0, transitType='secondary') if include_eclipse else 1.0
    
    phase_curve_model = models.phase_curve_func(model_params, times, init_t0) if include_phase_curve else 1.0
    
    # non-systematics model (i.e. (star + planet) / star
    physical_model = transit_model*eclipse_model*line_model*phase_curve_model
    
    # compute the systematics model
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "Invalid method."
    residuals = fluxes / physical_model
    sensitivity_map = models.compute_sensitivity_map(model_params=model_params, method=method, xcenters=xcenters, ycenters=ycenters, 
                                                        residuals=residuals, knots=knots, nearIndices=nearIndices, xBinSize=x_bin_size, 
                                                        yBinSize=y_bin_size, ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree, 
                                                        pld_intensities=pld_intensities, model=physical_model)
    
    model = physical_model*sensitivity_map
    
    output = {}
    output['full_model'] = model
    # output['line_model'] = line_model
    output['physical_model'] = physical_model
    output['sensitivity_map'] = sensitivity_map
    
    print('LOADING...')
    
    return output
