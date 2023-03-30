import batman
import emcee
import joblib
import numpy as np
import pandas as pd
import os

import corner
try:
    import pinknoise
    HAS_PINKNOISE = True
except ImportError:
    print("Please install `pinknoise` before importing")
    HAS_PINKNOISE = False

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from exomast_api import exoMAST_API
from matplotlib import pyplot as plt
# from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.special import gammaln, ndtri
from statsmodels.robust import scale
from time import time
from tqdm import tqdm

from wanderer import load_wanderer_instance_from_file
# from wanderer.utils import plotly_scattergl_flux_over_time

from skywalker import krdata  # , bliss, pld, utils
from skywalker.utils import (
    output_data_from_file,
    identify_outliers,
    configure_krdata_noise_models
)

from multiprocessing import Pool, cpu_count


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
    aornums: np.ndarray = None
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


@dataclass
class ExoplanetTSOData:
    times: np.ndarray = None
    fluxes: np.ndarray = None
    flux_errs: np.ndarray = None
    aornums: np.ndarray = None
    npix: np.ndarray = None
    pld_intensities: np.ndarray = None
    xcenters: np.ndarray = None
    ycenters: np.ndarray = None
    xwidths: np.ndarray = None
    ywidths: np.ndarray = None


class SpitzerKRDataEmcee:

    def __init__(
            self, df=None, aor_dir=None, channel=None, planet_name=None,
            trim_size=0, timebinsize=0, mast_name=None, n_samples=1000,
            inj_fpfs=0, nwalkers=32, centering_key=None, aper_key=None,
            init_fpfs=None, estimate_pinknoise=False, n_piecewise_params=0,
            n_sig=5, process_mcmc=False, run_full_pipeline=False, savenow=False, visualise_mle=False, visualise_chains=False, visualise_mcmc_results=False, standardise_fluxes=False, standardise_times=False, standardise_centers=False, verbose=False):

        self.df = df
        self.aor_dir = aor_dir
        self.channel = channel
        self.planet_name = planet_name
        self.mast_name = mast_name
        self.centering_key = centering_key
        self.aper_key = aper_key
        self.n_samples = n_samples
        self.nwalkers = nwalkers
        self.estimate_pinknoise = estimate_pinknoise and HAS_PINKNOISE
        self.n_piecewise_params = n_piecewise_params
        self.inj_fpfs = inj_fpfs
        self.init_fpfs = init_fpfs
        self.process_mcmc = process_mcmc
        self.savenow = savenow
        self.visualise_mle = visualise_mle
        self.visualise_mcmc_results = visualise_mcmc_results
        self.visualise_chains = visualise_chains
        self.trim_size = trim_size
        self.timebinsize = timebinsize
        self.centering_key = centering_key
        self.aper_key = aper_key
        self.n_sig = n_sig
        self.standardise_fluxes = standardise_fluxes
        self.standardise_times = standardise_times
        self.standardise_centers = standardise_centers
        self.verbose = verbose

        if self.mast_name is None:
            self.mast_name = self.planet_name

        if run_full_pipeline:
            self.full_pipeline()

    def initialise_data_and_params(self):
        # create self.tso_data
        self.preprocess_pipeline()

        start_krdata = time()
        self.configure_krdata(ymod=0.7, n_nbr=100)
        end_krdata = time()

        if self.verbose:
            print(f'KRData Creation took {end_krdata - start_krdata} seconds')

        self.configure_planet_info()
        self.initialize_fit_params()
        self.initialize_bm_params()

        if self.estimate_pinknoise:
            self.configure_pinknoise_model()

        if self.inj_fpfs > 0:
            self.inject_eclipse()

    def initialize_bm_params(self):
        # object to store transit parameters
        self.bm_params = batman.TransitParams()
        self.bm_params.per = self.period  # orbital period
        self.bm_params.t0 = self.tcenter  # time of inferior conjunction
        self.bm_params.inc = self.inc  # inclunaition in degrees

        # semi-major axis (in units of stellar radii)
        self.bm_params.a = self.aprs

        # planet radius (in units of stellar radii)
        self.bm_params.rp = self.rprs
        self.bm_params.ecc = self.ecc  # eccentricity
        self.bm_params.w = self.omega  # longitude of periastron (in degrees)
        self.bm_params.limb_dark = self.ldtype  # limb darkening model

        if self.ecenter is None:
            self.ecenter = self.tcenter + 0.5 * self.period

        if self.ldtype == 'uniform':
            self.bm_params.u = []  # limb darkening coefficients

        elif self.ldtype == 'linear':
            self.bm_params.u = [self.u1]  # limb darkening coefficients

        elif self.ldtype == 'quadratic':
            # limb darkening coefficients
            self.bm_params.u = [self.u1, self.u2]
        else:
            raise ValueError(
                "`ldtype` can only be ['uniform', 'linear', 'quadratic']")

        self.bm_params.t_secondary = self.ecenter
        self.bm_params.fp = self.fpfs

    def full_pipeline(self):
        self.initialise_data_and_params()

        self.run_mle_pipeline()

        if self.process_mcmc:
            self.run_emcee_pipeline()

        if self.visualise_mle:
            visualise_mle_solution(self)

        if self.visualise_chains:
            visualise_emcee_traces_corner(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

        if self.visualise_mcmc_results:
            visualise_emcee_samples(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

        if self.savenow:
            self.save_mle_emcee()

    def initialize_fit_params(self, init_logf=-5.0):
        # Compute MLE
        np.random.seed(42)

        init_fpfs = self.fpfs  # 265 ppm
        # init_ecenter = self.ecenter0  # t0 + per/2

        self.init_params = [
            np.random.normal(init_fpfs, 1e-5),
            np.random.normal(0.0, 1e-3),
            np.random.normal(init_logf, 0.01)
        ]

        if self.estimate_pinknoise:
            self.init_params.extend(
                np.random.normal([0.5]*2, [1e-4]*2)
            )

        if 0 < self.n_piecewise_params <= 2:
            # Add a offset and slope for each AOR
            self.add_piecewise_linear(
                n_lines=0,
                add_slope=self.n_piecewise_params == 2  # 2 params
            )

    def inject_eclipse(self):
        ppm = 1e6

        print(f'Injecting Model with FpFs: {self.inj_fpfs*ppm}ppm')

        # Inject a signal if `inj_fpfs` is provided
        inj_model = self.batman_wrapper(
            self.tso_data.times,
            period=self.period,
            tcenter=self.tcenter,
            inc=self.inc,
            aprs=self.aprs,
            rprs=self.rprs,
            ecc=self.ecc,
            omega=self.omega,
            u1=self.u1,
            u2=self.u2,
            offset=self.offset,
            slope=self.slope,
            curvature=self.curvature,
            ecenter=self.ecenter,
            fpfs=self.inj_fpfs,
            ldtype=self.ldtype,  # ='uniform',
            transit_type=self.transit_type,  # ='secondary',
            verbose=self.verbose
        )

        # print(fluxes.mean(), inj_model.min(), inj_model.max())
        self.fluxes = self.fluxes * inj_model

        self.tso_data.fluxes = self.fluxes
        self.krdata_inputs.fluxes = self.fluxes

    def run_mle_pipeline(self, init_fpfs=None):

        if init_fpfs is not None:
            self.init_params[0] = np.random.normal(init_fpfs, 1e-5)

        # nll = lambda *args: -self.log_emcee_likelihood(*args)
        nlp = lambda *args: -self.log_emcee_probability(*args)

        print('init_params:', self.init_params)
        if self.verbose:
            print('init_params:', self.init_params)

        self.soln = minimize(nlp, self.init_params)  # , args=())

        if self.verbose:
            print(
                f'fpfs_ml={self.soln.x[0]*1e6}\n'
                f'delta_ecenter_ml={self.soln.x[1]}\n'
                f'ecenter_ml={self.ecenter0 + self.soln.x[1]}\n'
                f'log_f_ml={self.soln.x[3]}\n'
            )

        # Plot MLE Results
        # x0 = np.linspace(0, 10, 500)

    def run_emcee_pipeline(self, alpha=1e-4):
        # Emcee Sampling
        init_distro = alpha * np.random.randn(self.nwalkers, len(self.soln.x))

        # Ensure that "we" are all on the same page
        self.bm_params.fp = self.soln.x[0]

        pos = self.soln.x + init_distro
        nwalkers, ndim = pos.shape

        # Avoid complications with MKI
        os.environ["OMP_NUM_THREADS"] = "1"

        # with Pool(cpu_count()-1) as pool:
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_emcee_probability,  # pool=pool
        )

        start = time()
        self.sampler.run_mcmc(pos, self.n_samples, progress=True)
        print(f"Multiprocessing took { time() - start:.1f} seconds")

    def preprocess_pipeline(self):

        if self.trim_size > 0:
            df = trim_initial_timeseries(
                df=self.df,
                trim_size=self.trim_size,
                aornums=self.df.aornum.unique()
            )

        if self.timebinsize > 0:
            med_df, std_df = bin_df_time(self.df, timebinsize=self.timebinsize)

            # Option 1
            self.df = med_df.copy()

            """
            # Option 2
            self.df = med_df.copy()
            for colname in df.columns:
                if 'noise' in colname:
                    std_colname = colname.replace('noise', 'flux')
                    self.df[colname] = std_df[std_colname]
            """

            del med_df, std_df

        if self.df is None:
            tso_data = load_from_wanderer(
                planet_name=self.planet_name,
                channel=self.channel,
                aor_dir=self.aor_dir,
                aper_key='gaussian_fit_annular_mask_rad_2.5_0.0',
                centering_key=self.centering_key
            )
        else:
            tso_data = load_from_df(
                self.df,
                aper_key=self.aper_key,
                centering_key=self.centering_key
            )

        isfinite = np.isfinite(tso_data.times)
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.fluxes))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.flux_errs))
        # isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.aornums))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.ycenters))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.xcenters))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.npix))

        times = tso_data.times[isfinite]
        fluxes = tso_data.fluxes[isfinite]
        flux_errs = tso_data.flux_errs[isfinite]
        aornums = tso_data.aornums[isfinite]
        ycenters = tso_data.ycenters[isfinite]
        xcenters = tso_data.xcenters[isfinite]
        npix = tso_data.npix[isfinite]

        med_flux = np.median(fluxes)
        flux_errs = flux_errs / med_flux
        fluxes = fluxes / med_flux

        arg_times = times.argsort()
        fluxes = fluxes[arg_times]
        flux_errs = flux_errs[arg_times]
        aornums = aornums[arg_times]
        times = times[arg_times]
        ycenters = ycenters[arg_times]
        xcenters = xcenters[arg_times]
        npix = npix[arg_times]

        if self.standardise_centers:
            # Center by assuming eclipse is near center
            ycenter = (ycenter - ycenter.mean()) / ycenter.std()
            xcenter = (xcenter - xcenter.mean()) / xcenter.std()

        if self.standardise_times:
            # Center by assuming eclipse is near center
            times = times - times.mean()

        if self.standardise_fluxes:
            # Center by assuming eclipse is near center
            med_flux = np.median(fluxes)
            std_flux = scale.mad(fluxes)

            idxkeep = np.abs(fluxes - med_flux) < self.n_sig * std_flux

        self.tso_data = ExoplanetTSOData(
            times=times[idxkeep],
            fluxes=fluxes[idxkeep],
            flux_errs=flux_errs[idxkeep],
            aornums=aornums[idxkeep],
            ycenters=ycenters[idxkeep],
            xcenters=xcenters[idxkeep],
            npix=npix[idxkeep]
        )

        # # TODO: Confirm if this is still required
        # self.tso_data.times = self.tso_data.times
        # self.tso_data.fluxes = self.tso_data.fluxes
        # self.tso_data.flux_errs = self.tso_data.flux_errs
        # self.tso_data.aornums = self.tso_data.aornums

    def print_bm_params(self):
        print('times', self.tso_data.times.mean())
        print('period', self.period)
        print('tcenter', self.tcenter)
        print('inc', self.inc)
        print('aprs', self.aprs)
        print('rprs', self.rprs)
        print('ecc', self.ecc)
        print('omega', self.omega)
        print('u1', self.u1)
        print('u2', self.u2)
        print('offset', self.offset)
        print('slope', self.slope)
        print('curvature', self.curvature)
        print('ldtype', self.ldtype)
        print('transit_type', self.transit_type)
        print('t_secondary', self.ecenter)
        print('fp', self.fpfs)

    def batman_wrapper(self, update_fpfs=None, update_ecenter=None):
        '''
            Written By Jonathan Fraine
            https://github.com/exowanderer/Fitting-Exoplanet-Transits
        '''
        # print(fpfs, ecenter)
        if self.verbose >= 2:
            self.print_bm_params()

        if update_fpfs is not None:
            self.bm_params.fp = update_fpfs

        if update_ecenter is not None:
            self.bm_params.t_secondary = update_ecenter

        m_eclipse = batman.TransitModel(
            self.bm_params,
            self.tso_data.times,
            transittype=self.transit_type
        )

        return m_eclipse.light_curve(self.bm_params)

    def batman_krdata_wrapper(self, update_fpfs=None, update_ecenter=None):
        '''
            Written By Jonathan Fraine
            https://github.com/exowanderer/Fitting-Exoplanet-Transits
        '''
        batman_curve = self.batman_wrapper(
            update_fpfs=update_fpfs,
            update_ecenter=update_ecenter
        )

        # Gaussian Kernel Regression Spitzer Sensivity map
        krdata_map = krdata.sensitivity_map(
            self.krdata_inputs.fluxes / batman_curve,  # OoT_curvature
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        # TODO: Check is the OoT should should be multiplied or added
        return batman_curve * krdata_map  # + OoT_curvature

    """
    def batman_spitzer_wrapper(self):
        return self.batman_krdata_wrapper(
            times=self.tso_data.times,
            krdata_inputs=self.krdata_inputs,
            period=self.period,
            tcenter=self.tcenter,
            ecenter=self.ecenter,
            inc=self.inc,
            aprs=self.aprs,
            rprs=self.rprs,
            fpfs=self.fpfs,
            ecc=self.ecc,
            omega=self.omega,
            u1=self.u1,
            u2=self.u2,
            offset=self.offset,
            slope=self.slope,
            curvature=self.curvature,
            ldtype=self.ldtype,
            transit_type=self.transit_type
        )
    """

    def piecewise_linear_model(self, theta):
        times = self.tso_data.times.copy()
        aornums = self.tso_data.aornums.copy()
        sorted_aornums = np.sort(np.unique(aornums))

        offsets = theta[3::2]
        slopes = theta[4::2]

        # print(offsets, slopes)
        piecewise_line = np.zeros_like(times)
        for offset_, slope_, aornum_ in zip(offsets, slopes, sorted_aornums):
            is_aornum = aornums == aornum_
            line_ = linear_model(times[is_aornum], offset_, slope_)
            piecewise_line[is_aornum] = line_
            # print(aornum_, offset_, slope_, is_aornum.sum(), line_.mean())

        return piecewise_line

    def piecewise_offset_model(self, theta):
        times = self.tso_data.times.copy()
        aornums = self.tso_data.aornums.copy()
        sorted_aornums = np.sort(np.unique(aornums))

        offsets = theta[3:]

        # print(offsets, slopes)
        piecewise_offset = np.zeros_like(times)
        for offset_, aornum_ in zip(offsets, sorted_aornums):
            is_aornum = aornums == aornum_
            line_ = linear_model(times[is_aornum], offset_, 0)
            piecewise_offset[is_aornum] = line_
            # print(aornum_, offset_, slope_, is_aornum.sum(), line_.mean())

        return piecewise_offset

    def log_emcee_prior(self, theta):
        # return 0  # Open prior
        fpfs, delta_ecenter, log_f = theta[:3]  #
        sigma_w, sigma_r = theta[3:5] if self.estimate_pinknoise else 1, 1
        gamma = theta[5] if len(theta) == 6 else 1.0

        # print('log_prior', fpfs, delta_ecenter, log_f)  #
        ppm = 1e6
        # TODO: Experiment with negative eclipse depths
        #   to avoid non-Gaussian posterior distributions
        ed_min = 0 / ppm  # Explore existance by permitting non-physical distributions
        ed_max = 500 / ppm  # 500 ppm
        dc_max = 0.05  # day
        dc_min = -0.05  # day

        lf_min = -10  # 1e-10 error modifier
        lf_max = 1  # 10x error modifier

        sw_min = 0
        sr_min = 0
        gm_min = 1.0

        return (
            0.0 if (
                ed_min <= fpfs < ed_max and
                dc_min < delta_ecenter < dc_max and
                lf_min < log_f < lf_max and
                sw_min <= sigma_w and
                sr_min <= sigma_r and
                gm_min <= gamma
            )
            else -np.inf
        )
    """
    def log_emcee_likelihood(self, theta):
        fpfs, delta_ecenter, log_f = theta[:3]  #
        # print(fpfs, self.fpfs)
        # self.fpfs = fpfs
        # self.ecenter = self.ecenter0  # + delta_ecenter

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(theta)
        # pw_line = self.piecewise_offset_model(theta)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        vector_likelihood = -0.5 * residuals ** 2 / sigma2 + np.log(sigma2)

        if self.estimate_pinknoise and len(theta) >= 5:
            if len(theta[3:]) == 5:
                sigma_w, sigma_r = theta[3:]
                gamma = 1
            if len(theta[3:]) == 6:
                sigma_w, sigma_r, gamma = theta[3:]

            self.wavelet_model.get_likelihood(residuals, sigma_w, sigma_r)
            self.wavelet_model.get_likelihood(
                residuals, sigma_w, sigma_r, gamma=gamma
            )
            wavelet_log_likelihood = 0
            vector_likelihood += wavelet_log_likelihood

        return np.sum(vector_likelihood)
    """

    def get_wavelet_log_likelihood(self, residuals, theta):
        if len(theta[3:]) == 2:
            sigma_w, sigma_r = theta[3:]
            gamma = 1.0
        if len(theta[3:]) == 3:
            sigma_w, sigma_r, gamma = theta[3:]

        if residuals.size != self.ndata_wavelet:
            # Pad up to self.ndata_wavelet
            padding = np.zeros(self.ndata_wavelet - residuals.size)
            residuals = np.r_[residuals, padding]

        # print(residuals.sum(), sigma_w, sigma_r, gamma)
        return self.wavelet_model.get_likelihood(
            residuals,
            sigma_w,
            sigma_r,
            gamma=gamma
        )

    def log_emcee_likelihood(self, theta):
        fpfs, delta_ecenter, log_f = theta[:3]  #
        # print(fpfs, self.fpfs)
        # self.fpfs = fpfs
        # self.ecenter = self.ecenter0  # + delta_ecenter

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(theta)
        # pw_line = self.piecewise_offset_model(theta)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        residuals = (fluxes - model)
        wavelet_log_likelihood = 0
        if self.estimate_pinknoise and len(theta) >= 5:
            wavelet_log_likelihood = -0.5*self.get_wavelet_log_likelihood(
                residuals=fluxes - model,
                theta=theta
            )

        normal_log_likelihood = residuals ** 2 / sigma2 + np.log(sigma2)
        normal_log_likelihood = -0.5 * np.sum(normal_log_likelihood)

        if wavelet_log_likelihood != 0 and self.verbose:
            print('normal_log_likelihood', normal_log_likelihood)
            print('wavelet_log_likelihood', wavelet_log_likelihood)

        return normal_log_likelihood + wavelet_log_likelihood

    def log_emcee_probability(self, theta):
        lp = self.log_emcee_prior(theta)

        return (
            lp + self.log_emcee_likelihood(theta)
            if np.isfinite(lp) else -np.inf
        )

    def configure_krdata(self, ymod=0.7, n_nbr=100):

        self.noise_model_config = configure_krdata_noise_models(
            self.tso_data.ycenters.copy(),
            self.tso_data.xcenters.copy(),
            self.tso_data.npix.copy(),
            ymod=ymod,
            n_nbr=n_nbr
        )

        self.krdata_inputs = KRDataInputs(
            ind_kdtree=self.noise_model_config.ind_kdtree,
            gw_kdtree=self.noise_model_config.gw_kdtree,
            fluxes=self.tso_data.fluxes
        )

    def configure_planet_info(self, init_fpfs=None, init_u1=0, init_u2=0):

        if init_fpfs is None:
            init_fpfs = self.init_fpfs

        if init_fpfs is None:
            ppm = 1e6
            init_fpfs = 265 / ppm

        self.planet_info = exoMAST_API(planet_name=self.mast_name)

        planet_info = self.planet_info  # save white space below

        planet_info.Rp_Rs = planet_info.Rp_Rs or None  # for later

        if not hasattr(planet_info, 'Rp_Rs') or planet_info.Rp_Rs is None:
            print('[WARNING] Rp_Rs does not exist in `planet_info`')
            print('Assuming Rp_Rs == sqrt(transit_depth)')
            planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

        self.period = planet_info.orbital_period
        self.tcenter = planet_info.transit_time
        self.inc = planet_info.inclination
        self.aprs = planet_info.a_Rs
        self.tdepth = planet_info.transit_depth
        self.rprs = planet_info.Rp_Rs
        self.ecc = planet_info.eccentricity
        self.omega = planet_info.omega
        self.u1 = init_u1
        self.u2 = init_u2
        self.offset = 1
        self.slope = 0
        self.curvature = 0
        self.ldtype = 'uniform'
        self.transit_type = 'secondary'

        init_ecenter = self.tcenter + self.period*0.5

        self.ecenter = init_ecenter
        self.ecenter0 = init_ecenter
        self.fpfs = init_fpfs

    def configure_pinknoise_model(self):
        ndata = len(self.tso_data.times)
        power_of_two = np.log(ndata) / np.log(2)
        if power_of_two != int(power_of_two):
            power_of_two = np.ceil(power_of_two)

        self.ndata_wavelet = 2**power_of_two

        lrg_pow2 = np.log(self.ndata_wavelet) / np.log(2)

        assert (lrg_pow2 == int(lrg_pow2)), \
            'Wavelet requires power of two data points'

        self.ndata_wavelet = int(self.ndata_wavelet)
        self.wavelet_model = pinknoise.compute(self.ndata_wavelet)

    def save_mle_emcee(self, filename=None):
        discard = 0
        thin = 1
        self.tau = []

        try:
            self.tau = self.sampler.get_autocorr_time()
        except Exception as err:
            print(err)

        emcee_output = {
            'flat_samples': self.sampler.get_chain(
                discard=discard, thin=thin, flat=True
            ),
            'samples': self.sampler.get_chain(),
            'tau': self.tau
        }

        isotime = datetime.now(timezone.utc).isoformat()
        if filename is None:
            filename = f'emcee_spitzer_krdata_ppm_results_{isotime}.joblib.save'

        joblib.dump({'emcee': emcee_output, 'mle': self.soln}, filename)
        print(f'Saved emcee and mle results to {filename}')

    def load_mle_emcee(self, filename=None, isotime=None):

        assert (filename is not None or isotime is None), \
            'Please provide either `filename` or `isotime`'

        if isotime is not None:
            if filename is not None:
                print(
                    Warning(
                        '`isotime` is not None. '
                        'Therfore `filename` will be overwritten'
                    )
                )

            filename = f'emcee_spitzer_krdata_ppm_results_{isotime}.joblib.save'

        results = joblib.load(filename)

        self.flat_samples = results['emcee']['flat_samples']
        self.samples = results['emcee']['samples']
        self.tau = results['emcee']['tau']


def grab_data_from_csv(filename=None, aornums=None):
    if filename is None:
        filename = 'ExtractedData/ch2/hatp26b_ch2_complete_median_full_df.csv'

    df0 = pd.read_csv(filename)

    if aornums is None:
        aornums = df0.aornum.unique()

    hours2days = 24
    n_trim = 1.0 / hours2days
    df_sub_list = []
    for aor_ in aornums:
        df_aor_ = df0.query(f'aornum == "{aor_}"')

        trim_start = df_aor_.bmjd_adjstd.iloc[0] + n_trim
        df_aor_ = df_aor_.query(f'bmjd_adjstd > {trim_start}')

        df_sub_list.append(df_aor_)

    df = pd.concat(df_sub_list)

    # Sort by combined BMJD-ajstd without dropping BMJD-ajstd
    df = df.set_index(keys='bmjd_adjstd', drop=False)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def bin_df_time(df, timebinsize=0):

    if timebinsize <= 0:
        return

    med_binned = {}
    std_binned = {}

    for aor_ in tqdm(df.aornum.unique(), desc='AORs'):
        df_aor_ = df.query(f'aornum == "{aor_}"')

        time_range = df_aor_.bmjd_adjstd.max() - df_aor_.bmjd_adjstd.min()
        nbins_aor_ = np.ceil(time_range / timebinsize).astype(int)

        start = df_aor_.bmjd_adjstd.min()
        end = start + timebinsize
        med_binned[aor_] = []
        std_binned[aor_] = []
        for _ in tqdm(range(nbins_aor_), desc='Bins'):
            end = start + timebinsize
            df_in_bin = df_aor_.query(f'{start} < bmjd_adjstd < {end}')
            med_binned[aor_].append(df_in_bin.median(numeric_only=True))

            unc_ = df_in_bin.std(numeric_only=True) / np.sqrt(len(df_in_bin))
            std_binned[aor_].append(unc_)

            # print(k, start, end, df.bmjd_adjstd.max(), df_in_bin)
            start = end

        med_binned[aor_] = pd.concat(med_binned[aor_], axis=1).T
        std_binned[aor_] = pd.concat(std_binned[aor_], axis=1).T

        med_binned[aor_]['aornum'] = aor_
        std_binned[aor_]['aornum'] = aor_

    med_binned = pd.concat(med_binned.values())
    std_binned = pd.concat(std_binned.values())

    med_binned = med_binned.set_index(keys='bmjd_adjstd', drop=False)
    med_binned.sort_index(inplace=True)
    med_binned.reset_index(inplace=True, drop=True)

    std_binned = std_binned.set_index(keys='bmjd_adjstd', drop=False)
    std_binned.sort_index(inplace=True)
    std_binned.reset_index(inplace=True, drop=True)

    return med_binned, std_binned


def trim_initial_timeseries(df, trim_size=0, aornums=None):

    if trim_size <= 0:
        return df

    if aornums is None:
        aornums = df.aornum.unique()

    trimmed = []
    for aor_ in aornums:
        df_aor_ = df.query(f'aornum == "{aor_}"')
        bmjd_min = df_aor_.bmjd_adjstd.min()
        trimmed_ = df_aor_.query(f'bmjd_adjstd >= {bmjd_min + trim_size}')
        trimmed.append(trimmed_)

    df_trimmed = pd.concat(trimmed, axis=0)

    # Sort by combined BMJD-ajstd without dropping BMJD-ajstd
    df_trimmed = df_trimmed.set_index(keys='bmjd_adjstd', drop=False)
    df_trimmed.sort_index(inplace=True)
    df_trimmed.reset_index(inplace=True, drop=True)

    return df_trimmed


def batman_wrapper(
        times, period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2,
        offset, slope, curvature, ecenter=None, fpfs=None,
        ldtype='uniform', transit_type='secondary', verbose=False):
    '''
        Written By Jonathan Fraine
        https://github.com/exowanderer/Fitting-Exoplanet-Transits
    '''
    # print(fpfs, ecenter)
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

    if ecenter is None:
        ecenter = tcenter + 0.5 * period

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

    bm_params.t_secondary = ecenter
    bm_params.fp = fpfs

    m_eclipse = batman.TransitModel(
        bm_params,
        times,
        transittype=transit_type
    )

    return m_eclipse.light_curve(bm_params)


def batman_plotting_wrapper(spitzer_analysis, fpfs=0, delta_ecenter=0):
    return batman_wrapper(
        times=spitzer_analysis.tso_data.times,
        period=spitzer_analysis.period,
        tcenter=spitzer_analysis.tcenter,
        ecenter=spitzer_analysis.ecenter0+delta_ecenter,
        inc=spitzer_analysis.inc,
        aprs=spitzer_analysis.aprs,
        rprs=spitzer_analysis.rprs,
        fpfs=fpfs,
        ecc=spitzer_analysis.ecc,
        omega=spitzer_analysis.omega,
        u1=spitzer_analysis.u1,
        u2=spitzer_analysis.u2,
        offset=spitzer_analysis.offset,
        slope=spitzer_analysis.slope,
        curvature=spitzer_analysis.curvature,
        ldtype=spitzer_analysis.ldtype,
        transit_type=spitzer_analysis.transit_type
    )


def add_piecewise_linear(init_params, n_lines=0, add_slope=False):
    # Add a offset and slope for each AOR
    offset_default = 1.0
    slope_default = 0.0
    # for _ in np.sort(tso_data.aornums.unique()):
    for _ in range(n_lines):
        offset_ = np.random.normal(offset_default, 1e-4)
        if add_slope:
            slope_ = np.random.normal(slope_default, 1e-5)
            init_params.extend([offset_, slope_])
        else:
            init_params.append(offset_)

    return init_params


def linear_model(times, offset, slope):
    times = times.copy() - times.mean()
    return offset + times * slope


def print_mle_results(soln_x, ecenter0=0):

    fpfs_ml, delta_ecenter_ml, log_f_ml = soln_x[:3]
    if len(soln_x) >= 5:
        sigma_w, sigma_r = soln_x[3:5]
        if len(soln_x) == 6:
            gamma = soln_x[6]

    print(
        f'fpfs_ml={fpfs_ml*1e6:.3f} ppm\n'
        f'delta_ecenter_ml={delta_ecenter_ml:.5f}'
    )
    if ecenter0 > 0:
        print(
            f'ecenter_ml={ecenter0 + delta_ecenter_ml:.3f}'
        )

    print(
        f'log_f_ml={log_f_ml:.3f}'
    )

    if len(soln_x) >= 5:
        print(
            f'sigma_w={sigma_w:.3f}\n'
            f'sigma_r={sigma_r:.3f}'
        )
        if len(soln_x) == 6:
            print(
                f'gamma={gamma:.3f}\n'
            )


def load_from_df(df, aper_key=None, centering_key=None):
    if aper_key is None:
        aper_key = 'rad_2p5_0p0'

    if centering_key is None:
        centering_key = 'fluxweighted'

    times = df.bmjd_adjstd.values
    fluxes = df[f'flux_{aper_key}'].values
    flux_errs = df[f'noise_{aper_key}'].values
    aornums = df['aornum'].values
    ycenters = df[f'{centering_key}_ycenters'].values
    xcenters = df[f'{centering_key}_xcenters'].values
    npix = df.effective_widths

    # Confirm npix is a DataFrame, Series, or NDArray
    npix = npix.values if hasattr(df.effective_widths, 'values') else npix

    return ExoplanetTSOData(
        times=times,
        fluxes=fluxes,
        flux_errs=flux_errs,
        aornums=aornums,
        ycenters=ycenters,
        xcenters=xcenters,
        npix=npix
    )


def load_from_wanderer(
        planet_name, channel, aor_dir, aper_key=None, centering_key=None):

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
    aornums = np.array([aor_dir] * times.size)
    ycenters = wanderer_.centering_df[f'{centering_key}_ycenters'].values
    xcenters = wanderer_.centering_df[f'{centering_key}_xcenters'].values
    npix = wanderer_.effective_widths

    return ExoplanetTSOData(
        times=times,
        fluxes=fluxes,
        flux_errs=flux_errs,
        aornums=aornums,
        ycenters=ycenters,
        xcenters=xcenters,
        npix=npix
    )


def get_labels(ndim):

    labels = ["fpfs", "delta-ecenter", "log(f)"]  #
    if ndim >= 5:
        labels.extend(['sigma_w', 'sigma_r'])

    if ndim == 6:
        labels.append('gamma')

    return labels


def trace_plot(sampler):
    # Compute Estimators
    samples = sampler.get_chain()
    n_chain_samples, nwalkers, ndim = samples.shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    n_chain_burning = int(burnin * n_chain_samples)
    samples = samples.copy()
    labels = get_labels(ndim)

    print(labels)
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

    # for i in range(ndim):
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")


def flatten_chains(sampler, discard=100, thin=15, burnin=0.2):
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    n_flat_samples, ndim = flat_samples.shape

    n_flat_burnin = int(burnin * n_flat_samples)
    return flat_samples.copy()[n_flat_burnin:]


def print_emcee_results(sampler, discard=100, thin=15, burnin=0.2):
    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    _, ndim = flat_samples.shape
    labels = get_labels(ndim)

    for i in range(len(labels)):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        if labels[i] == 'fpfs':
            mcmc[1] = mcmc[1]*1e6
            q = q * 1e6

        txt = f"{labels[i]} = {mcmc[1]:.4f}_-{q[0]:.4f}^{q[1]:.4f}"
        print(txt)


def get_truth_emcee_values(sampler, discard=100, thin=15, burnin=0.2):
    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    _, ndim = flat_samples.shape

    # TODO: Add upper and lower quantils into Corner plot labels
    perctiles = [16, 50, 84]
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], perctiles)
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], perctiles)
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], perctiles)

    so_called_truth = [fpfs_mcmc*1e6, delta_ecenter_mcmc, log_f_mcmc]  #

    if ndim >= 5:
        _, sigma_w_mcmc, _ = np.percentile(flat_samples[:, 3], perctiles)
        _, sigma_r_mcmc, _ = np.percentile(flat_samples[:, 4], perctiles)

        so_called_truth.extend([sigma_w_mcmc, sigma_r_mcmc])

    if ndim == 6:
        _, gamma_mcmc, _ = np.percentile(flat_samples[:, 5], perctiles)
        so_called_truth.append(gamma_mcmc)

    return so_called_truth


# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
def visualise_emcee_traces_corner(
        spitzer_analysis, discard=100, thin=15, burnin=0.2, verbose=False):

    # Save White Space Below
    # times = spitzer_analysis.tso_data.times
    # fluxes = spitzer_analysis.tso_data.fluxes
    # yerr = spitzer_analysis.tso_data.flux_errs
    sampler = spitzer_analysis.sampler
    # Convert from decimal to ppm

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    n_flat_samples, ndim = flat_samples.shape

    try:
        tau = sampler.get_autocorr_time()
        print(f'tau: {tau}')
    except Exception as err:
        print(err)

    print_emcee_results(sampler, discard=100, thin=15, burnin=0.2)

    trace_plot(sampler)
    # Display Corner Plots
    so_called_truth = get_truth_emcee_values(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )
    fpfs_mcmc, delta_ecenter_mcmc, log_f_mcmc = so_called_truth[:3]
    if ndim >= 5:
        sigma_w_mcmc, sigma_r_mcmc = so_called_truth[3:5]
    if ndim == 6:
        gamma_mcmc = so_called_truth[6]

    if verbose:
        print(
            f'fpfs_ml={fpfs_mcmc*1e6}\n'
            f'delta_ecenter_ml={delta_ecenter_mcmc}\n'
            f'ecenter_ml={spitzer_analysis.ecenter0 + delta_ecenter_mcmc}\n'
            f'log_f_ml={log_f_mcmc}\n'
        )
        if ndim >= 5:
            print(
                f'log_f_ml={sigma_w_mcmc}\n'
                f'log_f_ml={sigma_r_mcmc}\n'
            )
        if ndim == 6:
            print(
                f'log_f_ml={gamma_mcmc}\n'
            )

    ppm = 1e6
    labels = get_labels
    flat_samples[:, 0] = flat_samples[:, 0] * ppm
    fig = corner.corner(
        flat_samples[:, :len(labels)],
        labels=labels,
        truths=so_called_truth,
        show_titles=True
    )

    plt.show()


def visualise_emcee_samples(
        spitzer_analysis, discard=100, thin=15, burnin=0.2, verbose=False):

    # Save White Space Below
    times = spitzer_analysis.tso_data.times
    fluxes = spitzer_analysis.tso_data.fluxes
    yerr = spitzer_analysis.tso_data.flux_errs
    sampler = spitzer_analysis.sampler

    samples = sampler.get_chain()
    n_chain_samples, nwalkers, ndim = samples.shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    n_flat_samples, ndim = flat_samples.shape

    n_flat_burnin = int(burnin * n_flat_samples)

    n_chain_burning = int(burnin * n_chain_samples)

    # Convert from decimal to ppm
    flat_samples = flat_samples.copy()[n_flat_burnin:]
    samples = samples.copy()

    # TODO: Add upper and lower quantils into Corner plot labels
    perctiles = [16, 50, 84]
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], perctiles)
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], perctiles)
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], perctiles)

    if ndim >= 5:
        _, sigma_w_mcmc, _ = np.percentile(flat_samples[:, 3], perctiles)
        _, sigma_r_mcmc, _ = np.percentile(flat_samples[:, 4], perctiles)

    if ndim == 6:
        _, gamma_mcmc, _ = np.percentile(flat_samples[:, 5], perctiles)

    """
    zorder = 3*times.size+1
    pw_line = piecewise_linear_model(soln.x, spitzer_analysis)
    plt.plot(
        times,
        pw_line,
        "-",
        color="violet",
        label="Piecewiese Linear",
        lw=3,
        zorder=zorder
    )
    """

    tcenter = spitzer_analysis.tcenter
    period = spitzer_analysis.period

    n_epochs = (times.min() - tcenter) / period
    n_epochs = np.ceil(n_epochs)
    n_shifts = n_epochs*spitzer_analysis.period

    labels = ["fpfs", "delta_ecenter", "log(f)"]  # "ecenter",
    """
    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples[:, :len(labels)]), size=100)

    for ind in inds:
        fpfs_, _ = flat_samples[ind, :len(labels)]  # delta_ecenter_,
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                spitzer_analysis,
                fpfs_,
                delta_ecenter_,
            ),
            "C1",
            alpha=0.1,
            # label="MCMC Estimator",
            lw=3,
            zorder=3*times.size+1
        )

    # n_epochs = 695.0
    ecenter_mcmc = spitzer_analysis.ecenter0 + n_shifts  # + delta_ecenter_mcmc
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_mcmc,
            delta_ecenter_,
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
    """
    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples[:, :len(labels)]), size=100)
    for ind in inds:
        fpfs_, delta_ecenter_ = flat_samples[ind, :len(labels)]
        ecenter_ = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                spitzer_analysis,
                fpfs_,
                delta_ecenter_,
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

    ecenter_mcmc = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_mcmc
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_mcmc,
            delta_ecenter_mcmc,
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


def visualise_mle_solution(spitzer_analysis):
    soln = spitzer_analysis.soln
    fpfs_ml, delta_ecenter_ml, log_f_ml, *offs_slopes = soln.x  #
    # pw_line = piecewise_linear_model(soln.x, spitzer_analysis)
    # pw_line = piecewise_offset_model(soln.x, spitzer_analysis)

    # n_epochs = 695.0
    n_epochs = (spitzer_analysis.tso_data.times.min() -
                spitzer_analysis.tcenter) / spitzer_analysis.period
    n_epochs = np.ceil(n_epochs)

    n_shifts = n_epochs*spitzer_analysis.period

    times = spitzer_analysis.tso_data.times
    fluxes = spitzer_analysis.tso_data.fluxes
    flux_errs = spitzer_analysis.tso_data.flux_errs
    zorder = 3*times.size+1

    ecenter_ml = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_ml
    plt.errorbar(times, fluxes, yerr=flux_errs, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_ml,
            delta_ecenter_ml,
        ),
        "-",
        color="orange",
        label="ML",
        lw=3,
        zorder=zorder
    )
    """
    plt.plot(
        times,
        pw_line,
        "-",
        color="violet",
        label="Piecewiese Linear",
        lw=3,
        zorder=zorder
    )
    """

    plt.axvline(
        ecenter_ml,
        color='green',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2,
        label='Eclipse Center'
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()


def plot_phase_data_by_aor(df, planet_name, init_fpfs=None):

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
        ppm = 1e6
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

    for k, aor_ in enumerate(df.aornum.unique()):
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


def trapezoid_transit(time, f, df, p, tt, tf=None, off=0, square=False):
    """
    Flux, from a uniform star source with single orbiting planet, as a function of time
    :param time: 1D array, input times
    :param f: unobscured flux, max flux level
    :param df: ratio of obscured to unobscured flux
    :param p: period of planet's orbit
    :param tt: total time of transit
    :param tf: time during transit in which flux doesn't change
    :param off: time offset. A value of 0 means the transit begins immediately
    :param square: If True, the shape of the transit will be square (tt == tf)
    :return: 1D array, flux from the star
    """
    if tf is None:
        tf = tt
    if tt <= tf:
        # Default to square shaped transit
        square = True

    y = []
    if not square:
        # define slope of sides of trapezium
        h = f*df*tt/(tt-tf)
        grad = 2*h/tt

    for i in time:
        j = (i + off) % p
        if j < tt:
            # transit
            # square shaped transit
            if square:
                y.append(f*(1 - df))

            # trapezium shaped transit
            elif j/tt < 0.5:
                # first half of transit
                val = f - grad*j
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
            else:
                # last half of transit
                val = (grad*j) - 2*h + f
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
        else:
            # no transit
            y.append(f)
    return y


def loglike_trapezoid(theta, fluxes, times):
    """
    Function to return the log likelihood of the trapezium shpaed transit light curve model
    :param theta: tuple or list containing each parameter
    :param fluxes: list or array containing the observed flux of each data point
    :param times: list or array containing the times at which each data point is recorded
    """
    # unpack parameters
    f_like, df_like, p_like, tt_like, tf_like, off_like = theta
    # expected value
    lmbda = np.array(
        trapezoid_transit(
            times, f_like, df_like, p_like, tt_like, tf_like, off=off_like
        )
    )

    # n = len(fluxes)
    a = np.sum(gammaln(np.array(fluxes)+1))
    b = np.sum(np.array(fluxes) * np.log(lmbda))

    return -np.sum(lmbda) - a + b


def prior_transform_trapezoid(theta, priors):
    """
    Transforms parameters from a unit hypercube space to their true space
    for a trapezium transit model
    """
    params = [0 for _ in range(len(theta))]
    for i in range(len(theta)):
        if i == 0:
            # uniform transform for f
            params[i] = (priors[i][1]-priors[i][0])*theta[i] + priors[i][0]
        else:
            # normal transform for remaining parameters
            params[i] = priors[i][0] + priors[i][1]*ndtri(theta[i])

    return np.array(params)


def get_prior_trapezoid():
    # uniform prior on flux
    f_min = 4.9
    f_max = 5.8

    # normal prior on flux drop
    df_mu = 0.19
    df_sig = 0.005

    # normal prior on period
    p_mu = 0.8372
    p_sig = 0.008

    # normal prior on total transit time
    tt_mu = 0.145
    tt_sig = 0.01

    # normal prior on flat transit time
    tf_mu = 0.143
    tf_sig = 0.01

    # normal prior on offset
    off_mu = 0.1502
    off_sig = 0.0008

    return [
        (f_min, f_max),
        (df_mu, df_sig),
        (p_mu, p_sig),
        (tt_mu, tt_sig),
        (tf_mu, tf_sig),
        (off_mu, off_sig)
    ]

    # remove tf for square transit parameters
    # priors_square = priors[:4] + priors[5:]

    # return priors_trapezo
