'''
SKYWALKER: Spitzer Key Yield With All that Lightcurve Exoplanet Research.
Written by Carlos E. Munoz-Romero and Jonathan D. Fraine, 2018.
'''

# Import internal dependencies and methods.
from . import bliss
from . import krdata as kr
from . import utils
from . import models
from .models import line_model_func, trapezoid_model, transit_model_func
from .models import phase_curve_func

# Import external dependencies and methods.
import numpy as np
import batman
import corner
import argparse
# import exoparams
import json
import matplotlib
import matplotlib.pyplot as plt

from exomast_api import exoMAST_API
from os import environ
from pandas import DataFrame
from scipy import special
from scipy.interpolate import CubicSpline
from sklearn.externals import joblib
from functools import partial
from lmfit import Parameters, Minimizer, report_errors
from scipy import spatial
from scipy.interpolate import CubicSpline
from statsmodels.robust import scale
from time import time

# Global constants.
y, x = 0, 1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0
# Function definitions


def add_cubicspline_to_phase_curve_model(model_params, times, init_t0,
                                         phase_curve_model, eclipse_model):

    ecl_bottom = eclipse_model == eclipse_model.min()
    in_eclipse = eclipse_model < eclipse_model.max()

    eclipse0 = in_eclipse * (times < times.mean())
    eclipse1 = in_eclipse * (times > times.mean())

    t1_0 = np.where(eclipse0)[0][0]
    t2_0 = np.where(eclipse0 * ecl_bottom)[0][0]
    t3_0 = np.where(eclipse0 * ecl_bottom)[0][-1]
    t4_0 = np.where(eclipse0)[0][-1]

    t1_1 = np.where(eclipse1)[0][0]
    t2_1 = np.where(eclipse1 * ecl_bottom)[0][0]
    t3_1 = np.where(eclipse1 * ecl_bottom)[0][-1]
    t4_1 = np.where(eclipse1)[0][-1]

    y1_0 = phase_curve_model[t1_0]
    y2_0 = 1  # - model_params['edepth'].value
    y3_0 = 1  # - model_params['edepth'].value
    y4_0 = phase_curve_model[t4_0]

    y1_1 = phase_curve_model[t1_1]
    y2_1 = 1  # - model_params['edepth'].value
    y3_1 = 1  # - model_params['edepth'].value
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

    output_model[t2_0:t3_0] = y2_0  # == 1.0
    output_model[t2_1:t3_1] = y2_1  # == 1.0

    cs_local = CubicSpline(times[cs_idx], output_model[cs_idx])
    print("THIS IS BROKEN")
    return cs_local(times, 1)


def add_trap_to_phase_curve_model(model_params, times, init_t0,
                                  phase_curve_model, eclipse_model):

    ecl_bottom = eclipse_model == eclipse_model.min()
    in_eclipse = eclipse_model < eclipse_model.max()

    eclipse0 = in_eclipse * (times < times.mean())
    eclipse1 = in_eclipse * (times > times.mean())

    t1_0 = np.where(eclipse0)[0][0]
    t2_0 = np.where(eclipse0 * ecl_bottom)[0][0]
    t3_0 = np.where(eclipse0 * ecl_bottom)[0][-1]
    t4_0 = np.where(eclipse0)[0][-1]

    t1_1 = np.where(eclipse1)[0][0]
    t2_1 = np.where(eclipse1 * ecl_bottom)[0][0]
    t3_1 = np.where(eclipse1 * ecl_bottom)[0][-1]
    t4_1 = np.where(eclipse1)[0][-1]

    y1_0 = phase_curve_model[t1_0]
    y2_0 = 1  # - model_params['edepth'].value
    y3_0 = 1  # - model_params['edepth'].value
    y4_0 = phase_curve_model[t4_0]

    y1_1 = phase_curve_model[t1_1]
    y2_1 = 1  # - model_params['edepth'].value
    y3_1 = 1  # - model_params['edepth'].value
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

    output_model[t2_0:t3_0] = y2_0  # == 1.0
    output_model[t2_1:t3_1] = y2_1  # == 1.0

    output_model[t1_0:t2_0] = ingress_slope_0 * \
        (times[t1_0:t2_0] - x2_0) + y2_0
    output_model[t1_1:t2_1] = ingress_slope_1 * \
        (times[t1_1:t2_1] - x2_1) + y2_1
    output_model[t3_0:t4_0] = egress_slope_0 * (times[t3_0:t4_0] - x3_0) + y3_0
    output_model[t3_1:t4_1] = egress_slope_1 * (times[t3_1:t4_1] - x3_1) + y3_1

    return output_model


def instantiate_system(planet_input, fpfs=0.0,
                       u_params=[0.0, 0.0],
                       phase_curve_amp_1_0=0.0,
                       phase_curve_amp_1n1=0.0,
                       phase_curve_amp_1p1=0.0,
                       lmax=3, lambda0=90.0):
    """
        instantiate the planet :class:``Primary``,
        :class:``Secondary``, and :class:``System``
        with or without gradients.

    """
    try:
        from starry import kepler
    except:
        raise ImportError(
            '`starry` (github.com/rodluger/starry)'
            ' is required to run this operation.'
            ' Try `pip install starry`\n'
            '     `conda install starry`\n'
            ' or  `pip install git+https://github.com/rodluger/starry`'
        )

    if isinstance(planet_input, str):
        planet_info = exoMAST_API(planet_input)
    elif isinstance(planet_input, exoMAST_API):
        planet_info = planet_input
    else:
        raise TypeError('`planet_info` must have type of'
                        ' either `str` or `exoMAST_API`')

    # Instantiate the star
    star = kepler.Primary()

    star[1] = u_params[0]
    star[2] = u_params[1]

    # Instantiate the planet
    planet = kepler.Secondary(lmax=lmax)
    planet.lambda0 = lambda0  # Mean longitude in degrees at reference time

    if not hasattr(planet_info, 'Rp_Rs'):
        if planet_info.Rp_Rs is None:
            print('[WARNING] Rp_Rs does not exist in `planet_info`')
            print('Assuming Rp_Rs == sqrt(transit_depth)')
            planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

    planet.r = planet_info.Rp_Rs  # planetary radius in stellar radius
    planet.L = fpfs  # flux from planet relative to star
    planet.inc = planet_info.inclination  # orbital inclination
    planet.a = planet_info.a_Rs  # orbital distance in stellar radius
    planet.prot = planet_info.orbital_period  # synchronous rotation
    planet.porb = planet_info.orbital_period  # synchronous rotation
    planet.tref = planet_info.transit_time  # MJD for transit center time

    planet.ecc = planet_info.eccentricity  # eccentricity of orbit
    planet.Omega = planet_info.omega  # argument of the ascending node

    planet[1, 0] = phase_curve_amp_1_0  # Y_1_0
    planet[1, -1] = phase_curve_amp_1n1  # Y_1n1
    planet[1, 1] = phase_curve_amp_1p1  # Y_1p1

    # Instantiate the system
    system = kepler.System(star, planet)

    return star, planet, system


def update_starry_system(planet, star, system, model_params, times,
                         lambda0=90.0):

    star[1] = model_params['u1'].value
    star[2] = model_params['u2'].value

    rprs = np.sqrt(model_params['tdepth'].value)

    # Update the planetary parameters
    planet.lambda0 = lambda0  # Mean longitude in degrees at reference time
    planet.r = rprs  # planetary radius in stellar radius
    # flux from planet relative to star
    planet.L = model_params['edepth'].value
    planet.inc = model_params['inc'].value  # orbital inclination
    planet.a = model_params['aprs'].value  # orbital distance in stellar radius
    planet.prot = model_params['period'].value  # synchronous rotation
    planet.porb = model_params['period'].value  # synchronous rotation

    # MJD for transit center time
    planet.tref = model_params['init_t0'].value + model_params['deltaTc'].value

    planet.ecc = model_params['ecc'].value  # eccentricity of orbit
    # argument of the ascending node
    planet.Omega = model_params['omega'].value

    planet[1, -1] = model_params['Y_1n1'].value  # Sine Amplitude
    planet[1, 0] = model_params['Y_1_0'].value  # Cosine Amplitude
    planet[1, 1] = model_params['Y_1p1'].value  # Sine Amplitude

    system.compute(times)

    return system.lightcurve


def create_starry_lightcurve(planet, star, system, model_params, times,
                             lambda0=90.0):
    ''' Use the STARRY infrastucture to create a planetary phase curve model
    '''
    star[1] = model_params['u1'].value
    star[2] = model_params['u2'].value

    rprs = np.sqrt(model_params['tdepth'].value)

    # Update the planetary parameters
    planet.lambda0 = lambda0  # Mean longitude in degrees at reference time
    planet.r = rprs  # planetary radius in stellar radius
    # flux from planet relative to star
    planet.L = model_params['edepth'].value
    planet.inc = model_params['inc'].value  # orbital inclination
    planet.a = model_params['aprs'].value  # orbital distance in stellar radius
    planet.prot = model_params['period'].value  # synchronous rotation
    planet.porb = model_params['period'].value  # synchronous rotation

    # MJD for transit center time
    planet.tref = model_params['init_t0'].value + model_params['deltaTc'].value

    planet.ecc = model_params['ecc'].value  # eccentricity of orbit
    # argument of the ascending node
    planet.Omega = model_params['omega'].value

    # Double sigmoid model derived by Rodrigo Lugar for STARRY implementation
    #    Requires lmax == 3
    # planet_map[0, 0] = 1.0
    # planet_map[1, 0] = a_starry
    # planet_map[2, 0] = -a_starry * b_starry / np.sqrt(15)

    planet[1, -1] = model_params['Y_1n1'].value  # Sine Amplitude
    planet[1, 0] = model_params['Y_1_0'].value  # Cosine Amplitude
    planet[1, 1] = model_params['Y_1p1'].value  # Sine Amplitude

    system.compute(times)

    return system.lightcurve  # model_params['day2night'] *


def create_starry_lightcurve_double_sigmoid(planet, star, system, model_params, times,
                                            lambda0=90.0):
    ''' Use the STARRY infrastucture to create a planetary phase curve model
    '''
    star[1] = model_params['u1'].value
    star[2] = model_params['u2'].value

    rprs = np.sqrt(model_params['tdepth'].value)

    # Update the planetary parameters
    planet.lambda0 = lambda0  # Mean longitude in degrees at reference time
    planet.r = rprs  # planetary radius in stellar radius
    # flux from planet relative to star
    planet.L = model_params['edepth'].value
    planet.inc = model_params['inc'].value  # orbital inclination
    planet.a = model_params['aprs'].value  # orbital distance in stellar radius
    planet.prot = model_params['period'].value  # synchronous rotation
    planet.porb = model_params['period'].value  # synchronous rotation

    # MJD for transit center time
    planet.tref = model_params['init_t0'].value + model_params['deltaTc'].value

    planet.ecc = model_params['ecc'].value  # eccentricity of orbit
    # argument of the ascending node
    planet.Omega = model_params['omega'].value

    a_starry = model_params['a_starry']
    b_starry = model_params['b_starry']

    # Double sigmoid model derived by Rodrigo Lugar for STARRY implementation
    #    Requires lmax == 3
    planet_map[0, 0] = 1.0
    planet_map[1, 0] = a_starry
    planet_map[2, 0] = -a_starry * b_starry / np.sqrt(15)

    # planet[1,-1] = model_params['Y_1n1'].value # Sine Amplitude
    # planet[1, 0] = model_params['Y_1_0'].value # Cosine Amplitude
    # planet[1, 1] = model_params['Y_1p1'].value # Sine Amplitude

    system.compute(times)
    return model_params['day2night'] * system.lightcurve


def deltaphase_eclipse(ecc, omega):
    ''' Compute the delta phase offset for the eclipse relative to transit '''
    return 0.5 * (1 + (4. / np.pi) * ecc * np.cos(omega))


def find_eclipse_transits(times, model_params):
    ''' Find the index location of the eclipse and transit '''
    trantime = model_params['init_t0'] + model_params['deltaTc']
    per = model_params['period']
    eclphase = deltaphase_eclipse(model_params['ecc'], model_params['omega'])
    phase = ((times - trantime) % per) / per

    if (eclphase >= phase.min()) and (eclphase <= phase.max()):
        eclipse_loc = abs(phase - eclphase).argmin()
    else:
        eclipse_loc = None

    if (0.0 >= phase.min()) and (0.0 <= phase.max()):
        transit_loc = abs(phase).argmin()
    else:
        transit_loc = None

    return eclipse_loc, transit_loc


def generate_local_times(times, model_params,
                         eclipse_width=0.1, interp_ratio=0.1):
    ''' This generates smaller times array for interpolation with starry '''
    n_events = 3

    eclipse_loc, transit_loc = find_eclipse_transits(times, model_params)

    trantime = model_params['init_t0'] + model_params['deltaTc']
    per = model_params['period']

    phase = ((times - trantime) % per) / per

    if eclipse_loc is None:
        n_events -= 1

    if transit_loc is None:
        n_events -= 1

    assert(n_events > 0), "Error: Somehow the number of events is "\
                          "less than 0"

    n_pts_per_event = times.size * interp_ratio / n_events
    times_local = np.linspace(times.min(), times.max(), n_pts_per_event)
    times_local = list(times_local)

    if eclipse_loc is not None:
        eclipse_time = times[eclipse_loc]
        ecl_time_width = eclipse_width * per
        idx_eclipse_low = (
            times - (eclipse_time - 0.5 * ecl_time_width)).argmin()
        idx_eclipse_high = (
            times - (eclipse_time + 0.5 * ecl_time_width)).argmin()

        eclipse_times = np.linspace(times[idx_eclipse_low],
                                    times[idx_eclipse_high])

        times_local.extend(eclipse_times)

    if transit_loc is not None:
        transit_time = times[transit_loc]
        tr_time_width = transit_width * per
        idx_transit_low = (
            times - (transit_time - 0.5 * tr_time_width)).argmin()
        idx_transit_high = (
            times - (transit_time + 0.5 * tr_time_width)).argmin()

        transit_times = np.linspace(times[idx_transit_low],
                                    times[idx_transit_high])

        times_local.extend(transit_times)

    times_local = np.array(list(set(times_local)))

    return times_local[times_local.argsort()]


def compute_full_model_starry(model_params, times,  planet_info=None,
                              planet=None, star=None, system=None, lmax=3,
                              include_polynomial=True, return_case=None,
                              interpolate=False, interp_ratio=0.1,
                              eclipse_width=0.1, verbose=False):

    if interpolate:
        times_local = generate_local_times(times, model_params,
                                           eclipse_width=eclipse_width,
                                           interp_ratio=interp_ratio)
    else:
        times_local = times

    if None in [star, planet, system]:
        if planet_info is None:
            raise ValueError("Must provide [planet, star, system] "
                             "from `starry` or `planet_info` from "
                             " exoMAST_API to compute model.")

        star, planet, system = instantiate_system(planet_info, lmax=lmax)

    starry_model = create_starry_lightcurve(planet, star, system,
                                            model_params, times_local)

    if 'intercept' not in model_params.keys():
        include_polynomial = False

    line_model = models.line_model_func(model_params, times_local) \
        if include_polynomial else 1.0

    # non-systematics model (i.e. (star + planet) / star
    physical_model = line_model * starry_model

    if interpolate:
        physical_model_int = CubicSpline(times_local, physical_model)
        physical_model = physical_model_int(times)

    if return_case == 'dict':
        output = {}
        # output['line_model'] = line_model
        output['physical_model'] = physical_model
        output['line_model'] = line_model
        output['phase_curve_model'] = starry_model

        return output
    else:
        return physical_model


def compute_full_model_normal(model_params, times, include_transit=True,
                              include_eclipse=True, include_phase_curve=True,
                              include_polynomial=True,
                              eclipse_option='trapezoid',
                              subtract_edepth=True, return_case=None,
                              use_trap=False, verbose=False):

    init_t0 = model_params['tCenter']

    if 'tdepth' not in model_params.keys():
        include_transit = False
    if 'edepth' not in model_params.keys():
        include_eclipse = False
    if 'intercept' not in model_params.keys():
        include_polynomial = False
    if 'cosAmp' not in model_params.keys():
        include_phase_curve = False

    if include_polynomial:
        line_model = models.line_model_func(model_params, times)
    else:
        line_model = 1.0

    if include_transit:
        transit_model = models.transit_model_func(model_params, times,
                                                  init_t0, transitType='primary')
    else:
        transit_model = 1.0

    if include_eclipse:
        if use_trap:
            eclipse_model = models.trapezoid_model(
                model_params, times, init_t0)
        else:
            eclipse_model = models.transit_model_func(model_params, times,
                                                      init_t0, transitType='secondary')
    else:
        eclipse_model = 1.0

    if include_phase_curve:
        phase_curve_model = models.phase_curve_func(
            model_params, times, init_t0)
    else:
        phase_curve_model = 1.0

    if subtract_edepth:
        eclipse_model = eclipse_model - model_params['edepth'].value

    where_eclipse = np.where(eclipse_model < eclipse_model.max())[0]

    ecl_bottom = eclipse_model == eclipse_model.min()
    # model_params['edepth'].value = phase_curve_model[ecl_bottom].mean() - 1.0

    try:
        model_params['edepth'].value = phase_curve_model[
            ecl_bottom].max() - 1.0
    except:
        model_params['edepth'].value = 0.0

    mutl_ecl = True
    try:
        cond1 = model_params['edepth'].value > 0.0
        cond2 = np.isfinite(model_params['edepth'].value)
        if cond1 and cond2:
            mutl_ecl = False
            if eclipse_option == 'cubicspline':
                phase_curve_model = add_cubicspline_to_phase_curve_model(
                    model_params, times, init_t0,
                    phase_curve_model, eclipse_model)
            elif eclipse_option == 'trapezoid':
                phase_curve_model = add_trap_to_phase_curve_model(model_params,
                                                                  times, init_t0, phase_curve_model, eclipse_model)
        else:
            if verbose:
                print('Edepth: {}'.format(model_params['edepth'].value))
            mutl_ecl = False
    except Exception as e:
        mutl_ecl = False

        if verbose:
            print('\n[WARNING] Failure Occured with {}'.format(str(e)))
            print('[WARNING] Model Params at Failure were')
            for val in model_params.values():
                print('\t\t{:11}: {:3}\t[{:5}, {:5}]\t{}'.format(
                    val.name, str(val.value)[:10], str(val.min)[:5],
                    str(val.max)[:5], str(val.vary)))

            print('\n[WARNING] BATMAN Eclipse Model Mean: {}'.format(
                eclipse_model.mean()))

    # If the phase curve does exist (all == 1),
    #   then include the eclipse alongside the transit model
    try:
        if np.allclose(phase_curve_model, np.ones(phase_curve_model.size)):
            mutl_ecl = True
    except:
        mutl_ecl = False

    if eclipse_option == 'batman':
        mult_ecl = True

    # non-systematics model (i.e. (star + planet) / star
    physical_model = transit_model * line_model * phase_curve_model

    if mutl_ecl:
        physical_model = physical_model * eclipse_model

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


def compute_full_model(model_params, times, planet_info=None,
                       fit_function='starry', include_transit=True,
                       include_eclipse=True, include_phase_curve=True,
                       include_polynomial=True, eclipse_option='trapezoid',
                       interpolate=False, interp_ratio=0.1, subtract_edepth=True,
                       return_case=None, use_trap=False, verbose=False,
                       planet_input=None, planet=None,
                       star=None, system=None, lmax=3):

    if fit_function is 'starry':
        return compute_full_model_starry(model_params, times,
                                         planet_info=planet_info,
                                         planet=planet,
                                         star=star,
                                         system=system,
                                         lmax=lmax,
                                         include_polynomial=include_polynomial,
                                         return_case=return_case,
                                         interpolate=interpolate,
                                         interp_ratio=interp_ratio,
                                         verbose=verbose)

    if fit_function is 'normal':
        return compute_full_model_normal(model_params, times,
                                         # planet_info = planet_info,
                                         include_transit=include_transit,
                                         include_eclipse=include_eclipse,
                                         include_phase_curve=include_phase_curve,
                                         include_polynomial=include_polynomial,
                                         eclipse_option=eclipse_option,
                                         subtract_edepth=subtract_edepth,
                                         return_case=return_case,
                                         use_trap=use_trap,
                                         verbose=verbose)


def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs,
                   keep_inds, planet=None, star=None, system=None,
                   planet_info=None, knots=None, method=None, nearIndices=None,
                   ind_kdtree=None, gw_kdtree=None, pld_intensities=None,
                   x_bin_size=0.1, y_bin_size=0.1, transit_indices=None,
                   include_transit=True, include_eclipse=True,
                   include_phase_curve=True, include_polynomial=True,
                   testing_model=False, eclipse_option='trapezoid',
                   use_trap=False, interpolate=False, interp_ratio=0.1,
                   fit_function='starry', verbose=False):

    start = time()
    start0 = time()
    physical_model = compute_full_model(model_params, times,
                                        planet_info=planet_info,
                                        star=star,
                                        planet=planet,
                                        system=system,
                                        fit_function=fit_function,
                                        include_transit=include_transit,
                                        include_eclipse=include_eclipse,
                                        include_phase_curve=include_phase_curve,
                                        include_polynomial=include_polynomial,
                                        eclipse_option=eclipse_option,
                                        interpolate=interpolate,
                                        interp_ratio=interp_ratio,
                                        verbose=verbose)
    # print('Physical Model took {} seconds'.format(time() - start0))
    if testing_model:
        return physical_model

    # compute the systematics model
    assert ('bliss' in method.lower()
            or 'krdata' in method.lower()
            or 'pld' in method.lower()), "No valid method selected."

    residuals = fluxes / physical_model
    start0 = time()
    sensitivity_map = models.compute_sensitivity_map(
        model_params=model_params,
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

    weirdness = compute_weird_slope(model_params, times)

    model = physical_model * sensitivity_map * weirdness
    # print('Full Res Function took {} seconds'.format(time()-start))
    return (model - fluxes) / flux_errs


def residuals_func_multiepoch(model_params, times_list, xcenters_list,
                              ycenters_list, fluxes_list,
                              flux_errs_list, keep_inds_list,
                              knots_list=None, nearIndices_list=None,
                              ind_kdtree_list=None, gw_kdtree_list=None,
                              pld_intensities_list=None,
                              method=None, x_bin_size=0.1, y_bin_size=0.1,
                              transit_indices=None, include_transit=True,
                              include_eclipse=True,
                              include_phase_curve=True,
                              include_polynomial=True,
                              testing_model=False,
                              eclipse_option='trapezoid', use_trap=False,
                              verbose=False):

    model_params_single = model_params.copy()

    model_full = []
    fluxes_full = []
    flux_errs_full = []

    zippidy_do_dah = zip(times_list, xcenters_list, ycenters_list, fluxes_list,
                         flux_errs_list, keep_inds_list, knots_list,
                         nearIndices_list, ind_kdtree_list, gw_kdtree_list,
                         pld_intensities_list)

    for epoch, zippidy_day in enumerate(zippidy_do_dah):
        times, xcenters, ycenters, fluxes, flux_errs, keep_inds, knots, \
            nearIndices, ind_kdtree, gw_kdtree, pld_intensities = zippidy_day

        intercept_epoch = model_params_single['intercept{}'.format(epoch)]
        slope_epoch = model_params_single['slope{}'.format(epoch)]
        curvature_epoch = model_params_single['curvature{}'.format(epoch)]

        model_params_single['intercept'].value = intercept_epoch
        model_params_single['slope'].value = slope_epoch
        model_params_single['curvature'].value = curvature_epoch

        model_now = residuals_func(model_params_single, times, xcenters,
                                   ycenters, fluxes, flux_errs, keep_inds,
                                   knots=knots,
                                   method=method,
                                   nearIndices=nearIndices,
                                   ind_kdtree=ind_kdtree,
                                   gw_kdtree=gw_kdtree,
                                   pld_intensities=pld_intensities,
                                   x_bin_size=x_bin_size,
                                   y_bin_size=y_bin_size,
                                   transit_indices=transit_indices,
                                   include_transit=include_transit,
                                   include_eclipse=include_eclipse,
                                   include_phase_curve=include_phase_curve,
                                   include_polynomial=include_polynomial,
                                   testing_model=testing_model,
                                   eclipse_option=eclipse_option,
                                   use_trap=use_trap,
                                   verbose=verbose)

        model_full.extend(model_now)
        fluxes_full.extend(fluxes)
        flux_errs_full.extend(flux_errs)

    model_full = np.array(model_full)
    fluxes_full = np.array(fluxes_full)
    flux_errs_full = np.array(flux_errs_full)

    return (model_full - fluxes_full) / flux_errs_full


def map_fit_params(fit_params, fit_param_names, model_params):
    ''' A wrapper helper to convert the params from a scipy.optimize.minimize
            to a dictionary (i.e. LMFIT setup).
    '''
    assert(len(fit_params) == len(fit_param_names))

    model_params = model_params.copy()

    for p, pname in zip(fit_params, fit_param_names):
        model_params[pname].value = p

    return model_params


def chisq_func_scipy(fit_params, fit_param_names, model_params, times,
                     xcenters, ycenters, fluxes, flux_errs, knots, keep_inds,
                     method=None, nearIndices=None, ind_kdtree=None,
                     gw_kdtree=None, pld_intensities=None, x_bin_size=0.1,
                     y_bin_size=0.1, transit_indices=None,
                     include_transit=True, include_eclipse=True,
                     include_phase_curve=True, include_polynomial=True,
                     testing_model=False, eclipse_option='trapezoid',
                     use_trap=False, verbose=False):
    ''' A wrapper to convert the inputs from a scipy.optimize.minimize to a
            dictionary setup (i.e. LMFIT setup).
    '''

    # Convert scipy.optimize.minimize inputs to
    model_params_fit = map_fit_params(
        fit_params, fit_param_names, model_params)

    # compute CHI for CHISQ output
    weighted_residuals = residuals_func(model_params_fit, times, xcenters,
                                        ycenters, fluxes, flux_errs, knots, keep_inds,
                                        method=method,
                                        nearIndices=nearIndices,
                                        ind_kdtree=ind_kdtree,
                                        gw_kdtree=gw_kdtree,
                                        pld_intensities=pld_intensities,
                                        x_bin_size=x_bin_size,
                                        y_bin_size=y_bin_size,
                                        transit_indices=transit_indices,
                                        include_transit=include_transit,
                                        include_eclipse=include_eclipse,
                                        include_phase_curve=include_phase_curve,
                                        include_polynomial=include_polynomial,
                                        testing_model=testing_model,
                                        eclipse_option=eclipse_option,
                                        use_trap=use_trap,
                                        verbose=verbose)

    # Return the ChiSq output
    return np.sum(weighted_residuals**2.)


def compute_weird_slope(model_params, times, debug=False):
    if debug:
        print('before weirdslope')
    # If all 3 keys exists, then trigger weirdness vector
    weird_cond = True
    if debug:
        print(list(model_params.keys()))
    for key in ['t_start', 'weirdslope', 'weirdintercept']:
        weird_cond = weird_cond * key in model_params.keys()

    if debug:
        print('weirdslope flag: {}'.format(weird_cond))

    if weird_cond:
        if debug:
            print('IN weirdslope conditional')
        weirdness = np.ones(times.size)
        cond = times - times.mean() > model_params['t_start']
        if debug:
            print('t-start exists: {}'.format(cond.any()))
        if debug:
            print('there are {} weird elements'.format(len(times[cond])))
        cond_time_ = (times[cond] - times.mean())
        weirdslope_ = model_params['weirdslope']
        weirdintercept_ = model_params['weirdintercept']
        weirdness[cond] = weirdslope_ * cond_time_ + weirdintercept_
        if debug:
            print('there are {} weird elements'.format(len(weirdness[cond])))
        return weirdness

    if debug:
        print('THERE IS NO WEIRDNESS!!!')
    return 1.0


def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes,
                               knots, keep_inds, planet_info,
                               method=None, nearIndices=None, ind_kdtree=None,
                               gw_kdtree=None, pld_intensities=None,
                               x_bin_size=0.1, y_bin_size=0.1,
                               transit_indices=None, fit_function='starry',
                               interpolate=False, interp_ratio=0.1,
                               star=None, planet=None, system=None,
                               include_transit=True,
                               include_eclipse=True,
                               include_phase_curve=True,
                               include_polynomial=True,
                               eclipse_option='trapezoid',
                               verbose=False):

    output = compute_full_model(model_params, times,
                                planet_info=planet_info,
                                include_transit=include_transit,
                                include_eclipse=include_eclipse,
                                include_phase_curve=include_phase_curve,
                                include_polynomial=include_polynomial,
                                eclipse_option=eclipse_option,
                                interpolate=interpolate,
                                fit_function=fit_function,
                                planet=planet, star=star, system=system,
                                return_case='dict', verbose=verbose)

    # compute the systematics model
    assert ('bliss' in method.lower() or 'krdata' in method.lower()
            or 'pld' in method.lower()), "No valid method selected."

    residuals = fluxes / output['physical_model']
    sensitivity_map = models.compute_sensitivity_map(model_params=model_params,
                                                     method=method, xcenters=xcenters, ycenters=ycenters,
                                                     residuals=residuals, knots=knots, nearIndices=nearIndices,
                                                     xBinSize=x_bin_size, yBinSize=y_bin_size,
                                                     ind_kdtree=ind_kdtree, gw_kdtree=gw_kdtree,
                                                     pld_intensities=pld_intensities,
                                                     model=output['physical_model'])

    weirdness = compute_weird_slope(model_params, times)

    model = output['physical_model'] * sensitivity_map * weirdness

    output['full_model'] = model
    output['sensitivity_map'] = sensitivity_map
    output['weirdness'] = weirdness

    return output


def generate_best_fit_solution_scipy(fit_params, fit_param_names, model_params,
                                     times, xcenters, ycenters, fluxes, knots, keep_inds,
                                     method=None, nearIndices=None, ind_kdtree=None,
                                     gw_kdtree=None, pld_intensities=None, x_bin_size=0.1,
                                     y_bin_size=0.1, transit_indices=None,
                                     include_transit=True, include_eclipse=True,
                                     include_phase_curve=True, include_polynomial=True,
                                     eclipse_option='trapezoid', verbose=False):
    ''' A wrapper to convert the inputs from a scipy.optimize.minimize to a 
            dictionary setup (i.e. LMFIT setup).
    '''

    # wrap the inputs
    model_params_fit = map_fit_params(
        fit_params, fit_param_names, model_params)

    # call the dictionary based setup
    output = generate_best_fit_solution(model_params_fit, times, xcenters,
                                        ycenters, fluxes, knots, keep_inds,
                                        method=method,
                                        nearIndices=nearIndices,
                                        ind_kdtree=ind_kdtree,
                                        gw_kdtree=gw_kdtree,
                                        pld_intensities=pld_intensities,
                                        x_bin_size=x_bin_size,
                                        y_bin_size=y_bin_size,
                                        transit_indices=transit_indices,
                                        include_transit=include_transit,
                                        include_eclipse=include_eclipse,
                                        include_phase_curve=include_phase_curve,
                                        include_polynomial=include_polynomial,
                                        eclipse_option=eclipse_option,
                                        verbose=verbose)

    return output
