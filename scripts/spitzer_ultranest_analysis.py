from skywalker.utils import (
    output_data_from_file,
    identify_outliers,
    configure_krdata_noise_models
)
import batman
import joblib
import numpy as np
import pandas as pd
import os

import emcee
import corner

from dataclasses import dataclass
from datetime import datetime, timezone
from exomast_api import exoMAST_API
# from functools import partial
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from statsmodels.robust import scale
from time import time

from wanderer import load_wanderer_instance_from_file
# from wanderer.utils import plotly_scattergl_flux_over_time

from skywalker import krdata  # , bliss, pld, utils


def batman_wrapper(
        times, period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2,
        offset, slope, curvature, ecenter=None, fpfs=None,
        ldtype='uniform', transit_type='secondary', verbose=False):
    '''
        Written By Jonathan Fraine
        https://github.com/exowanderer/Fitting-Exoplanet-Transits
    '''

    if verbose:
        print('times', times.mean())
        print('period', period)
        print('tcenter', tcenter)
        print('inc', inc)
        print('aprs', aprs)
        print('rprs', rprs)
        print('ecc', ecc)
        print('omega', omega)
        print('u1', u1)
        print('u2', u2)
        print('offset', offset)
        print('slope', slope)
        print('curvature', curvature)
        print('ldtype', ldtype)
        print('transit_type', transit_type)
        print('t_secondary', ecenter)
        print('fp', fpfs)

    bm_params = batman.TransitParams()  # object to store transit parameters
    bm_params.per = period   # orbital period
    bm_params.t0 = tcenter  # time of inferior conjunction
    bm_params.inc = inc      # inclunaition in degrees
    bm_params.a = aprs     # semi-major axis (in units of stellar radii)
    bm_params.rp = rprs     # planet radius (in units of stellar radii)
    bm_params.ecc = ecc      # eccentricity
    bm_params.w = omega    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model

    if ldtype == 'uniform':
        bm_params.u = []  # limb darkening coefficients

    elif ldtype == 'linear':
        bm_params.u = [u1]  # limb darkening coefficients

    elif ldtype == 'quadratic':
        bm_params.u = [u1, u2]  # limb darkening coefficients
    else:
        raise ValueError(
            "`ldtype` can only be ['uniform', 'linear', 'quadratic']")

    bm_params.t_secondary = tcenter + period*0.5 if ecenter is None else ecenter
    bm_params.fp = 0 if fpfs is None else fpfs

    m_eclipse = batman.TransitModel(
        bm_params,
        times,
        transittype=transit_type
    )

    return m_eclipse.light_curve(bm_params)


def batman_krdata_wrapper(
        times, krdata_inputs, period, tcenter, inc, aprs, rprs, ecc, omega,
        u1, u2, offset, slope, curvature, ecenter=None, fpfs=None,
        ldtype='uniform', transit_type='secondary', deconstructed=False):
    '''
        Written By Jonathan Fraine
        https://github.com/exowanderer/Fitting-Exoplanet-Transits
    '''
    batman_curve = batman_wrapper(
        times=times,
        period=period,
        tcenter=tcenter,
        ecenter=ecenter,
        inc=inc,
        aprs=aprs,
        rprs=rprs,
        fpfs=fpfs,
        ecc=ecc,
        omega=omega,
        u1=u1,
        u2=u2,
        offset=offset,
        slope=slope,
        curvature=curvature,
        ldtype=ldtype,
        transit_type=transit_type
    )

    # Out of transit temporal model
    OoT_curvature = offset
    if slope:
        OoT_curvature = OoT_curvature + slope*(times - times.mean())
    if curvature:
        OoT_curvature = OoT_curvature + curvature*(times - times.mean())**2

    # Gaussian Kernel Regression Spitzer Sensivity map
    krdata_map = krdata.sensitivity_map(
        krdata_inputs.fluxes / batman_curve,  # OoT_curvature
        krdata_inputs.ind_kdtree,
        krdata_inputs.gw_kdtree
    )

    if deconstructed:
        return batman_curve, OoT_curvature, krdata_map

    # TODO: Check is the OoT should should be multiplied or added
    return batman_curve * krdata_map  # + OoT_curvature


def batman_emcee_wrapper(mcmc_args):
    return batman_krdata_wrapper(
        times=mcmc_args.times,
        krdata_inputs=mcmc_args.krdata_inputs,
        period=mcmc_args.period,
        tcenter=mcmc_args.tcenter,
        ecenter=mcmc_args.ecenter,
        inc=mcmc_args.inc,
        aprs=mcmc_args.aprs,
        rprs=mcmc_args.rprs,
        fpfs=mcmc_args.fpfs,
        ecc=mcmc_args.ecc,
        omega=mcmc_args.omega,
        u1=mcmc_args.u1,
        u2=mcmc_args.u2,
        offset=mcmc_args.offset,
        slope=mcmc_args.slope,
        curvature=mcmc_args.curvature,
        ldtype=mcmc_args.ldtype,
        transit_type=mcmc_args.transit_type,
        deconstructed=mcmc_args.deconstructed
    )


def batman_plotting_wrapper(fpfs, ecenter, mcmc_args):
    return batman_wrapper(
        times=mcmc_args.times,
        period=mcmc_args.period,
        tcenter=mcmc_args.tcenter,
        ecenter=ecenter,
        inc=mcmc_args.inc,
        aprs=mcmc_args.aprs,
        rprs=mcmc_args.rprs,
        fpfs=fpfs,
        ecc=mcmc_args.ecc,
        omega=mcmc_args.omega,
        u1=mcmc_args.u1,
        u2=mcmc_args.u2,
        offset=mcmc_args.offset,
        slope=mcmc_args.slope,
        curvature=mcmc_args.curvature,
        ldtype=mcmc_args.ldtype,
        transit_type=mcmc_args.transit_type
    )


def log_prior(theta):
    # return 0  # Open prior
    fpfs, delta_ecenter, log_f = theta
    # print('log_prior', fpfs, delta_ecenter, log_f)
    ppm = 1e6
    # TODO: Experiment with negative eclipse depths
    #   to avoid non-Gaussian posterior distributions
    ed_min = 0 / ppm  # Explore existance by permitting non-physical distributions
    ed_max = 500 / ppm  # 100000 ppm
    dc_min = -0.0  # day
    dc_max = 0.1  # day

    lf_min = -10  # 1e-10 error modifier
    lf_max = 1  # 10x error modifier

    if (
            ed_min <= fpfs < ed_max and
            dc_min < delta_ecenter < dc_max and
            lf_min < log_f < lf_max
    ):
        return 0.0

    # if outside uniform prior bounds
    return -np.inf


def log_likelihood(theta, mcmc_args):
    fpfs, delta_ecenter, log_f = theta
    # fpfs = 265/1e6  # Wallack et al 2019 value
    mcmc_args.fpfs = fpfs
    mcmc_args.ecenter = mcmc_args.ecenter0 + delta_ecenter

    flux_errs = mcmc_args.flux_errs
    fluxes = mcmc_args.fluxes

    model = batman_emcee_wrapper(mcmc_args)
    sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((fluxes - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, mcmc_args):
    lp = log_prior(theta)
    return lp+log_likelihood(theta, mcmc_args) if np.isfinite(lp) else -np.inf


@dataclass
class KRDataInputs:
    ind_kdtree: np.ndarray = None
    gw_kdtree: np.ndarray = None
    fluxes: np.ndarray = None


@dataclass
class MCMCArgs:
    times: np.ndarray = None
    fluxes: np.ndarray = None
    flux_errs: np.ndarray = None
    krdata_inputs: KRDataInputs = None
    period: float = 0
    tcenter: float = 0
    ecenter: float = 0
    ecenter0: float = 0  # made static to fit delta_ecenter
    inc: float = 0
    aprs: float = 0
    rprs: float = 0
    fpfs: float = 0
    ecc: float = 0
    omega: float = 0
    u1: float = 0
    u2: float = 0
    offset: float = 0
    slope: float = 0
    curvature: float = 0
    ldtype: str = 'uniform'
    transit_type: str = 'secondary'
    deconstructed: bool = False


def configure_krdata(fluxes, ycenters, xcenters, npix, ymod=0.7, n_nbr=100):
    noise_model_config = configure_krdata_noise_models(
        ycenters.copy(),
        xcenters.copy(),
        npix.copy(),
        ymod=ymod,
        n_nbr=n_nbr
    )

    return KRDataInputs(
        ind_kdtree=noise_model_config.ind_kdtree,
        gw_kdtree=noise_model_config.gw_kdtree,
        fluxes=fluxes
    )


def configure_mcmc_args(
        planet_name, tso_data, krdata_inputs, mast_name=None, init_fpfs=None):

    if mast_name is None:
        mast_name = planet_name

    planet_info = exoMAST_API(planet_name=mast_name)
    planet_info.Rp_Rs = planet_info.Rp_Rs or None  # for later
    if not hasattr(planet_info, 'Rp_Rs') or planet_info.Rp_Rs is None:
        print('[WARNING] Rp_Rs does not exist in `planet_info`')
        print('Assuming Rp_Rs == sqrt(transit_depth)')
        planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

    init_period = planet_info.orbital_period
    init_tcenter = planet_info.transit_time
    init_aprs = planet_info.a_Rs
    init_inc = planet_info.inclination

    init_tdepth = planet_info.transit_depth
    init_rprs = np.sqrt(init_tdepth)
    init_ecc = planet_info.eccentricity
    init_omega = planet_info.omega

    if init_fpfs is None:
        ppm = 1e6
        init_fpfs = 265 / ppm

    # init_ecenter = tso_data.times.mean()
    init_ecenter = init_tcenter + init_period*0.5
    print(init_fpfs, init_ecenter)
    init_u1 = 0
    init_u2 = 0
    init_offset = 1.0
    init_slope = 1e-10
    init_curvature = 1e-10

    ldtype = 'uniform'
    transit_type = 'secondary'
    deconstructed = False

    return MCMCArgs(
        times=tso_data.times,
        fluxes=tso_data.fluxes,
        flux_errs=tso_data.flux_errs,
        period=init_period,
        tcenter=init_tcenter,
        ecenter=init_ecenter,
        ecenter0=init_ecenter,
        inc=init_inc,
        aprs=init_aprs,
        rprs=init_rprs,
        fpfs=init_fpfs,
        ecc=init_ecc,
        omega=init_omega,
        u1=init_u1,
        u2=init_u2,
        offset=init_offset,
        slope=init_slope,
        curvature=init_curvature,
        ldtype=ldtype,
        transit_type=transit_type,
        deconstructed=deconstructed,
        krdata_inputs=krdata_inputs
    )


@dataclass
class ExoplanetTSOData:
    fluxes: np.ndarray = None
    times: np.ndarray = None
    flux_errs: np.ndarray = None
    npix: np.ndarray = None
    pld_intensities: np.ndarray = None
    xcenters: np.ndarray = None
    ycenters: np.ndarray = None
    xwidths: np.ndarray = None
    ywidths: np.ndarray = None


def load_from_df(df, aper_key=None, centering_key=None):
    if aper_key is None:
        aper_key = 'rad_2p5_0p0'

    if centering_key is None:
        centering_key = 'fluxweighted'

    times = df.bmjd_adjstd.values
    fluxes = df[f'flux_{aper_key}'].values
    flux_errs = df[f'noise_{aper_key}'].values
    ycenters = df[f'{centering_key}_ycenters'].values
    xcenters = df[f'{centering_key}_xcenters'].values
    npix = df.effective_widths

    # Confirm npix is a DataFrame, Series, or NDArray
    npix = npix.values if hasattr(df.effective_widths, 'values') else npix

    return times, fluxes, flux_errs, ycenters, xcenters, npix


def load_from_wanderer(planet_name, channel, aor_dir, aper_key=None):
    wanderer_, data_config = load_wanderer_instance_from_file(
        planet_name=planet_name,
        channel=channel,
        aor_dir=aor_dir,  # 'r64924928',  # 'r42621184',
        check_defaults=False,
        shell=False
    )

    if aper_key is not None:
        aper_key = 'gaussian_fit_annular_mask_rad_2.5_0.0'

    times = wanderer_.time_cube
    fluxes = wanderer_.flux_tso_df[aper_key].values
    flux_errs = wanderer_.noise_tso_df[aper_key].values
    ycenters = wanderer_.centering_df['fluxweighted_ycenters'].values
    xcenters = wanderer_.centering_df['fluxweighted_xcenters'].values
    npix = wanderer_.effective_widths

    return times, fluxes, flux_errs, ycenters, xcenters, npix


def preprocess_pipeline(
        df=None, aor_dir=None, channel=None, planet_name=None, mast_name=None,
        n_sig=5, standardise_fluxes=True, standardise_times=False,
        standardise_centers=False):

    if df is None:
        times, fluxes, flux_errs, ycenters, xcenters, npix = load_from_wanderer(
            planet_name=planet_name,
            channel=channel,
            aor_dir=aor_dir,
            aper_key='gaussian_fit_annular_mask_rad_2.5_0.0'
        )
    else:
        times, fluxes, flux_errs, ycenters, xcenters, npix = load_from_df(
            df,
            aper_key=None,
            centering_key=None
        )

    isfinite = np.isfinite(fluxes)
    isfinite = np.bitwise_and(isfinite, np.isfinite(flux_errs))
    isfinite = np.bitwise_and(isfinite, np.isfinite(times))
    isfinite = np.bitwise_and(isfinite, np.isfinite(ycenters))
    isfinite = np.bitwise_and(isfinite, np.isfinite(xcenters))
    isfinite = np.bitwise_and(isfinite, np.isfinite(npix))

    fluxes = fluxes[isfinite]
    flux_errs = flux_errs[isfinite]
    times = times[isfinite]
    ycenters = ycenters[isfinite]
    xcenters = xcenters[isfinite]
    npix = npix[isfinite]

    med_flux = np.median(fluxes)
    flux_errs = flux_errs / med_flux
    fluxes = fluxes / med_flux

    arg_times = times.argsort()
    fluxes = fluxes[arg_times]
    flux_errs = flux_errs[arg_times]
    times = times[arg_times]
    ycenters = ycenters[arg_times]
    xcenters = xcenters[arg_times]
    npix = npix[arg_times]

    if standardise_centers:
        # Center by assuming eclipse is near center
        ycenter = (ycenter - ycenter.mean()) / ycenter.std()
        xcenter = (xcenter - xcenter.mean()) / xcenter.std()

    if standardise_times:
        # Center by assuming eclipse is near center
        times = times - times.mean()

    if standardise_fluxes:
        # Center by assuming eclipse is near center
        med_flux = np.median(fluxes)
        std_flux = scale.mad(fluxes)

        idxkeep = np.abs(fluxes - med_flux) < n_sig * std_flux

    return ExoplanetTSOData(
        fluxes=fluxes[idxkeep],
        flux_errs=flux_errs[idxkeep],
        times=times[idxkeep],
        ycenters=ycenters[idxkeep],
        xcenters=xcenters[idxkeep],
        npix=npix[idxkeep]
    )


# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
def visualise_emcee_solution(
        sampler, mcmc_args, discard=100, thin=15, burnin=0.2, verbose=False):
    print(burnin)
    times = mcmc_args.times
    fluxes = mcmc_args.fluxes
    yerr = mcmc_args.flux_errs

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    n_flat_samples, ndim = flat_samples.shape

    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    n_chain_samples, nwalkers, ndim = samples.shape

    n_flat_burnin = int(burnin * n_flat_samples)
    n_chain_burning = int(burnin * n_chain_samples)

    print(flat_samples.shape, samples.shape)

    # Convert from decimal to ppm
    flat_samples = flat_samples.copy()[n_flat_burnin:]
    samples = samples.copy()  # [n_chain_burnin:]

    labels = ["fpfs", "delta_ecenter", "log(f)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    try:
        tau = sampler.get_autocorr_time()
        print(f'tau: {tau}')
    except Exception as err:
        print(err)

    # Compute Estimators

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = f"{labels[i]} = {mcmc[1]:.3f}_-{q[0]:.3f}^{q[1]:.3f}"
        print(txt)

    # Display Corner Plots

    # TODO: Add upper and lower quantils into Corner plot labels
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], [16, 50, 84])
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], [16, 50, 84])
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], [16, 50, 84])

    socalled_truth = [fpfs_mcmc, delta_ecenter_mcmc, log_f_mcmc]

    if verbose:
        print(
            f'fpfs_ml={fpfs_mcmc*1e6}\n'
            f'delta_ecenter_ml={delta_ecenter_mcmc}\n'
            f'ecenter_ml={mcmc_args.ecenter0 + delta_ecenter_mcmc}\n'
            f'log_f_ml={log_f_mcmc}\n'
        )

    fig = corner.corner(
        flat_samples, labels=labels, truths=socalled_truth
    )

    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples), size=100)

    for ind in inds:
        fpfs_, delta_ecenter_, _ = flat_samples[ind]
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                fpfs_,
                delta_ecenter_,
                mcmc_args
            ),
            "C1",
            alpha=0.1,
            # label="MCMC Estimator",
            lw=3,
            zorder=3*times.size+1
        )

    n_epochs = 695.0
    n_shifts = n_epochs*mcmc_args.period
    ecenter_mcmc = mcmc_args.ecenter0 + delta_ecenter_mcmc + n_shifts
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            fpfs_mcmc,
            ecenter_mcmc,
            mcmc_args
        ),
        "-",
        color="orange",
        label="MCMC Estimator",
        lw=3,
        zorder=3*times.size+1
    )
    plt.axvline(
        ecenter_mcmc,
        color='violet',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()

    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        fpfs_, delta_ecenter_, _ = flat_samples[ind]
        ecenter_ = mcmc_args.ecenter0 + delta_ecenter_ + n_shifts
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                fpfs_,
                ecenter_,
                mcmc_args
            ),
            "C1",
            alpha=0.1,
            # label="MCMC Estimator",
            lw=3,
            zorder=3*times.size+1
        )
        plt.axvline(
            ecenter_,
            color='violet',
            linewidth=1,
            alpha=0.1,
            zorder=3*times.size+2
        )

    ecenter_mcmc = mcmc_args.ecenter0 + delta_ecenter_mcmc + n_shifts
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            fpfs_mcmc,
            ecenter_mcmc,
            mcmc_args
        ),
        "-",
        color="orange",
        label="MCMC Estimator",
        lw=3,
        zorder=3*times.size+1
    )

    plt.axvline(
        ecenter_mcmc,
        color='violet',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()


def visualise_mle_solution(
        times, fluxes, yerr, fpfs_ml, delta_ecenter_ml, mcmc_args):

    n_epochs = 695.0
    n_shifts = n_epochs*mcmc_args.period

    ecenter_ml = mcmc_args.ecenter0 + delta_ecenter_ml + n_shifts
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            fpfs_ml,
            ecenter_ml,
            mcmc_args
        ),
        "-",
        color="orange",
        label="ML",
        lw=3,
        zorder=3*times.size+1
    )
    plt.axvline(
        ecenter_ml,
        color='violet',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()


def save_mle_mcmc(inj_fpfs, soln, sampler):
    discard = 0
    thin = 1
    tau = []

    try:
        tau = sampler.get_autocorr_time()
    except Exception as err:
        print(err)
    emcee_output = {
        'flat_samples': sampler.get_chain(
            discard=discard, thin=thin, flat=True
        ),
        'samples': sampler.get_chain(),
        'tau': tau
    }

    isotime = datetime.now(timezone.utc).isoformat()
    joblib.dump(
        {
            'emcee': emcee_output,
            'mle': soln
        },
        f'emcee_spitzer_krdata_injected_'
        f'{inj_fpfs}ppm_results_{isotime}.joblib.save'
    )


def main(
        df=None, aor_dir=None, channel=None, planet_name=None, mast_name=None,
        mcmc_args=None, n_samples=1000, inj_fpfs=0, init_fpfs=None, n_sig=5, runmcmc=True, savenow=True, visualise_mle=True, visualise_emcee=True,
        standardise_fluxes=True, standardise_times=True,
        standardise_centers=False, verbose=False):

    tso_data = preprocess_pipeline(
        df=df,
        aor_dir=aor_dir,
        channel=channel,
        planet_name=planet_name,
        mast_name=mast_name,
        n_sig=n_sig,
        standardise_fluxes=standardise_fluxes,
        standardise_times=standardise_times,
        standardise_centers=standardise_centers
    )

    times = tso_data.times
    fluxes = tso_data.fluxes
    flux_errs = tso_data.flux_errs
    ycenters = tso_data.ycenters
    xcenters = tso_data.xcenters
    npix = tso_data.npix

    if mcmc_args is None:
        start_krdata = time()
        krdata_inputs = configure_krdata(
            fluxes,
            ycenters,
            xcenters,
            npix,
            ymod=0.7,
            n_nbr=100
        )
        print(f'KRData Creation took {time() - start_krdata} seconds')

        mcmc_args = configure_mcmc_args(
            planet_name=planet_name,
            tso_data=tso_data,
            krdata_inputs=krdata_inputs,
            init_fpfs=init_fpfs,
            mast_name=mast_name
        )
    else:
        krdata_inputs = mcmc_args.krdata_inputs
        mcmc_args.times = tso_data.times
        mcmc_args.fluxes = tso_data.fluxes
        mcmc_args.flux_errs = tso_data.flux_errs

    # planet_info = exoMAST_API(planet_name=planet_name)
    # init_ecenter = planet_info.transit_time + planet_info.orbital_period*0.5

    # Compute MLE
    np.random.seed(42)

    init_f = 0.5
    init_fpfs = mcmc_args.fpfs  # 265 ppm
    init_ecenter = mcmc_args.ecenter0  # t0 + per/2

    init_params = [
        np.random.normal(init_fpfs, 1e-5),
        np.random.normal(0.0, 1e-3),
        np.random.normal(np.log(init_f), 0.01)
    ]

    print('HELLO')
    if inj_fpfs:
        ppm = 1e6
        print(f'Injecting Model with FpFs: {inj_fpfs*ppm}ppm')
        # Inject a signal if `inj_fpfs` is provided
        inj_model = batman_wrapper(
            times,
            period=mcmc_args.period,
            tcenter=mcmc_args.tcenter,
            inc=mcmc_args.inc,
            aprs=mcmc_args.aprs,
            rprs=mcmc_args.rprs,
            ecc=mcmc_args.ecc,
            omega=mcmc_args.omega,
            u1=mcmc_args.u1,
            u2=mcmc_args.u2,
            offset=mcmc_args.offset,
            slope=mcmc_args.slope,
            curvature=mcmc_args.curvature,
            ecenter=mcmc_args.ecenter,
            fpfs=inj_fpfs,
            ldtype='uniform',
            transit_type='secondary',
            verbose=False
        )
        # print(fluxes.mean(), inj_model.min(), inj_model.max())
        fluxes = fluxes * inj_model

        krdata_inputs.fluxes = fluxes
        mcmc_args.fluxes = fluxes
        mcmc_args.krdata_inputs = krdata_inputs

    # nll = lambda *args: -log_likelihood(*args)
    nlp = lambda *args: -log_probability(*args)

    print(init_params)
    soln = minimize(nlp, init_params, args=(mcmc_args))
    fpfs_ml, delta_ecenter_ml, log_f_ml = soln.x

    # import sys
    # sys.exit()

    if verbose:
        print(
            f'fpfs_ml={fpfs_ml*1e6}\n'
            f'delta_ecenter_ml={delta_ecenter_ml}\n'
            f'ecenter_ml={mcmc_args.ecenter0 + delta_ecenter_ml}\n'
            f'log_f_ml={log_f_ml}\n'
        )

    if visualise_mle:
        visualise_mle_solution(
            times, fluxes, flux_errs, fpfs_ml, delta_ecenter_ml, mcmc_args
        )

    # Plot MLE Results
    # x0 = np.linspace(0, 10, 500)

    if not runmcmc:
        return {}, soln, mcmc_args

    # Emcee Sampling
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=([mcmc_args])
    )

    # n_samples = 1000
    sampler.run_mcmc(pos, n_samples, progress=True)

    if visualise_emcee:
        visualise_emcee_solution(
            sampler,
            mcmc_args,
            discard=100,
            thin=15,
            burnin=0.2,
            verbose=verbose
        )

    if savenow:
        save_mle_mcmc(inj_fpfs, soln, sampler)

    return sampler, soln, mcmc_args


def plot_phase_data_by_aor(df, planet_name, init_fpfs=None):
    ppm = 1e6
    planet_info = exoMAST_API(planet_name=planet_name)

    init_period = planet_info.orbital_period
    init_tcenter = planet_info.transit_time
    init_aprs = planet_info.a_Rs
    init_inc = planet_info.inclination

    init_tdepth = planet_info.transit_depth
    init_rprs = np.sqrt(init_tdepth)
    init_ecc = planet_info.eccentricity
    init_omega = planet_info.omega

    if init_fpfs is None:
        init_fpfs = 265 / ppm

    init_ecenter = init_tcenter + init_period * 0.5

    init_u1 = 0
    init_u2 = 0
    init_offset = 1.0
    init_slope = 1e-10
    init_curvature = 1e-10

    ldtype = 'uniform'
    transit_type = 'secondary'
    verbose = False

    init_model = batman_wrapper(
        df.bmjd_adjstd.values,
        init_period,
        init_tcenter,
        init_inc,
        init_aprs,
        init_rprs,
        init_ecc,
        init_omega,
        init_u1,
        init_u2,
        init_offset,
        init_slope,
        init_curvature,
        ecenter=init_ecenter,
        fpfs=init_fpfs,
        ldtype=ldtype,
        transit_type=transit_type,
        verbose=verbose
    )

    phased = (df.bmjd_adjstd - init_tcenter) % init_period / init_period

    for k, aor_ in enumerate(np.unique(df.aornum)):
        df_aor_ = df.query(f'aornum == "{aor_}"')
        times_ = df_aor_.bmjd_adjstd
        phase_aor_ = (times_ - init_tcenter) % init_period / init_period
        plt.plot(phase_aor_, df_aor_.flux_rad_2p0_0p0+k*0.03, '.', label=aor_)
        plt.annotate(aor_, [phase_aor_.max()+0.005, 1+k*0.03])

    plt.tight_layout()

    plt.plot(phased, init_model, 'k.')

    plt.show()


def phase_bin_data(
        df, planet_name, n_phases=1000, min_phase=0.4596, max_phase=0.5942,
        keep_end=False):

    planet_info = exoMAST_API(planet_name=planet_name)
    init_period = planet_info.orbital_period
    init_tcenter = planet_info.transit_time

    phased = (df.bmjd_adjstd - init_tcenter) % init_period / init_period

    phase_bins = np.linspace(min_phase, max_phase, n_phases)

    phase_binned_flux = np.zeros_like(phase_bins)
    phase_binned_ferr = np.zeros_like(phase_bins)
    for k, (phstart, phend) in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
        in_phase = (phstart >= phased.values)*(phased.values <= phend)
        flux_in_phase = df.flux_rad_2p5_0p0.values[in_phase]
        phase_binned_flux[k] = np.median(flux_in_phase)
        phase_binned_ferr[k] = np.std(flux_in_phase)

    if keep_end:
        in_phase = phased.values >= phend
        flux_in_phase = df.flux_rad_2p5_0p0.values[in_phase]
        phase_binned_flux[-1] = np.median(flux_in_phase)
        phase_binned_ferr[-1] = np.std(flux_in_phase)
    else:
        phase_bins = phase_bins[:-1]
        phase_binned_flux = phase_binned_flux[:-1]
        phase_binned_ferr = phase_binned_ferr[:-1]

    return phase_bins, phase_binned_flux, phase_binned_ferr


if __name__ == '__main__':
    import pandas as pd
    from spitzer_analysis import (
        main,
        visualise_emcee_solution,
        save_mle_mcmc,
        # plot_phase_data_by_aor
    )

    ppm = 1e6
    n_sig = 5
    aor_dir = 'r64922368'
    channel = 'ch2'  # CHANNEL SETTING
    planet_name = 'hatp26b'
    mast_name = 'HAT-P-26b'
    inj_fpfs = 0 / ppm  # no injected signal
    init_fpfs = 265 / ppm  # no injected signal
    n_samples = 1000

    # df_r64922368 = pd.read_csv(
    #     f'ExtractedData/ch2/{aor_dir}/hatp26b_{aor_dir}_median_full_df.csv'
    # )

    df0 = pd.read_csv(
        'ExtractedData/ch2/hatp26b_ch2_complete_median_full_df.csv'
    )

    aornums = [
        # # 'r42621184',  # Passed Eclipse
        # # 'r42624768',  # Passed Eclipse
        # # 'r47023872',  # Transit
        'r50676480',
        'r50676736',
        'r50676992',
        'r50677248',
        'r64922368',
        'r64923136',
        'r64923904',
        'r64924672'
    ]

    df_sub_list = [df0.query(f'aornum == "{aornum_}"') for aornum_ in aornums]

    df = pd.concat(df_sub_list)

    # Sort by combined BMJD-ajstd without dropping BMJD-ajstd
    df = df.set_index(keys='bmjd_adjstd', drop=False)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True, drop=True)

    mcmc_args = None
    sampler, soln, mcmc_args = main(
        df=df,
        # aor_dir='r64922368',
        # channel='ch2',  # CHANNEL SETTING
        # planet_name='hatp26b',
        mcmc_args=mcmc_args,
        mast_name='HAT-P-26b',
        inj_fpfs=inj_fpfs,
        init_fpfs=init_fpfs,
        n_samples=n_samples,
        n_sig=n_sig,
        runmcmc=True,
        visualise_mle=True,
        visualise_emcee=True,
        savenow=True,
        standardise_fluxes=True,
        standardise_times=False,
        standardise_centers=False,
        verbose=True
    )

    savenow = True
    if savenow:
        save_mle_mcmc(inj_fpfs, soln, sampler)

    plotmore = True
    if plotmore:
        visualise_emcee_solution(
            sampler, mcmc_args, discard=100, thin=15, burnin=0.2
        )

        # phase_bins, phase_binned_flux, phase_binned_ferr = phase_bin_data(
        #     df,
        #     planet_name,
        #     n_phases=1000,
        #     min_phase=0.4596,
        #     max_phase=0.5942
        # )

        # plt.errorbar(phase_bins, phase_binned_flux, phase_binned_ferr, fmt='o')
