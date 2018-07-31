'''
SYNOD: Spitzer Systematcs Noise Decorrelation. Written by Carlos E. Munoz-Romero and Jonathan D. Fraine, 2018.
Usage: python synod.py -f photometry_file -p planet_params_file -m method -xb x_bin_size -yb y_bin_size
'''
# Import all dependencies and methods.
import numpy as np
import batman
import corner
import argparse
import exoparams
import json
import bliss
import krdata as kr
import utils
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
import models
# Global constants.
y,x = 0,1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0
# Function definitions

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, keep_inds, method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, pld_intensities=None, x_bin_size  = 0.1, y_bin_size  = 0.1, transit_indices=None):

    init_t0 = model_params['tCenter']
    ntransits = len(transit_indices)

    line_model = models.line_model_func(model_params, ntransits, transit_indices, times)
    transit_model = models.transit_model_func(model_params, times, init_t0, transitType='primary')
    eclipse_model = models.transit_model_func(model_params, times, init_t0, transitType='secondary') if 'edpeth' in model_params.keys() else 1.0

    # non-systematics model (i.e. (star + planet) / star
    model = transit_model*eclipse_model*line_model

    # compute the systematics model
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "Invalid method."
    residuals = fluxes/model

    sensitivity_map = models.compute_sensitivity_map(model_params=model_params, method=method, xcenters=xcenters,  ycenters=ycenters, residuals=residuals, knots=knots, nearIndices=nearIndices, xBinSize=x_bin_size, yBinSize=y_bin_size, ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree, pld_intensities=pld_intensities, model=model)

    model = model*sensitivity_map
    return (model - fluxes) / flux_errs

def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes, knots, keep_inds, method=None, nearIndices=None, ind_kdtree=None, gw_kdtree=None, pld_intensities=None, x_bin_size  = 0.1, y_bin_size  = 0.1, transit_indices=None):

    ntransits = len(transit_indices)
    init_t0 = model_params['tCenter']
    print('SYNOD has fitted {} transits or eclipses'.format(ntransits))
    line_model = models.line_model_func(model_params, ntransits, transit_indices, times)
    transit_model = models.transit_model_func(model_params, times, init_t0, transitType='primary')
    eclipse_model = models.transit_model_func(model_params, times, init_t0, transitType='secondary') if 'edpeth' in model_params.keys() else 1.0

    # non-systematics model (i.e. (star + planet) / star
    model = transit_model*eclipse_model*line_model


    # compute the systematics model
    assert (method.lower() == 'bliss' or method.lower() == 'krdata' or method.lower() == 'pld'), "Invalid method."
    residuals = fluxes/model
    sensitivity_map = models.compute_sensitivity_map(model_params=model_params, method=method, xcenters=xcenters, ycenters=ycenters, residuals=residuals, knots=knots, nearIndices=nearIndices, xBinSize=x_bin_size, yBinSize=y_bin_size, ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree, pld_intensities=pld_intensities, model=model)

    model = model*sensitivity_map

    output = {}
    output['full_model'] = model
    output['line_model'] = line_model
    output['transit_model'] = transit_model*eclipse_model
    output['sensitivity_map'] = sensitivity_map
    print('LOADING...')
    return output
