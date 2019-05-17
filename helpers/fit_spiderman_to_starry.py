from starry import kepler, Map
from pylab import *#; ion()
from pandas import DataFrame
from lmfit import report_errors
from scipy import optimize as op
from time import time

import numpy as np
import exomast_api # pip install git+https://github.com/exowanderer/exoMAST_API
import spiderman as sp
import corner
import skywalker

def generate_spiderman_model(times, planet_info, nrg_ratio, temp_night, 
							delta_T, T_star, spider_params, u1=0.1, u2=0.0, 
							lower_wavelength = 4.0, upper_wavelength = 5.0):
	
	planet_info.Rp_Rs = planet_info.Rp_Rs or None # for later
	if not hasattr(planet_info, 'Rp_Rs') or planet_info.Rp_Rs is None:
		print('[WARNING] Rp_Rs does not exist in `planet_info`')
		print('Assuming Rp_Rs == sqrt(transit_depth)')
		planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

	spider_params.t0 = planet_info.transit_time
	spider_params.per = planet_info.orbital_period

	spider_params.a_abs = planet_info.orbital_distance
	spider_params.inc = planet_info.inclination
	spider_params.ecc = planet_info.eccentricity
	spider_params.w = planet_info.omega
	spider_params.rp = planet_info.Rp_Rs
	spider_params.a = planet_info.a_Rs
	spider_params.p_u1 = u1
	spider_params.p_u2 = u2

	spider_params.xi = nrg_ratio
	spider_params.T_n = temp_night
	spider_params.delta_T = delta_T
	spider_params.T_s = T_star

	spider_params.l1 = lower_wavelength
	spider_params.l2 = upper_wavelength

	return spider_params.lightcurve(times)

def generate_starry_model(times, planet_info, fpfs, 
						lmax = 1, lambda0 = 90.0, Y_1_0 = 0.5):
	''' Instantiate Kepler STARRY model; taken from HD 189733b example'''
	# Star
	star = kepler.Primary()

	# Planet
	planet = kepler.Secondary(lmax=lmax)
	planet.lambda0 = lambda0 # Mean longitude in degrees at reference time

	planet_info.Rp_Rs = planet_info.Rp_Rs or None # for later
	if not hasattr(planet_info, 'Rp_Rs') or planet_info.Rp_Rs is None:
		print('[WARNING] Rp_Rs does not exist in `planet_info`')
		print('Assuming Rp_Rs == sqrt(transit_depth)')
		planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

	planet.r = planet_info.Rp_Rs # planetary radius in stellar radius
	planet.L = 0.0 # flux from planet relative to star
	planet.inc = planet_info.inclination # orbital inclination 
	planet.a = planet_info.a_Rs # orbital distance in stellar radius
	planet.prot = planet_info.orbital_period # synchronous rotation
	planet.porb = planet_info.orbital_period # synchronous rotation
	planet.tref = planet_info.transit_time # MJD for transit center time

	planet.ecc = planet_info.eccentricity # eccentricity of orbit
	planet.Omega = planet_info.omega # argument of the ascending node

	# System
	system = kepler.System(star, planet)
	# Instantiate the system
	system = kepler.System(star, planet)

	# Specific plottings
	

	# Blue Curve
	# NOTE: Prevent negative luminosity on the night side

	# Green Curve
	# Compute the normalization
	map = Map(1)
	map[0, 0] = 1
	map[1, 0] = Y_1_0
	norm = map.flux()

	planet.L = fpfs / norm
	planet[1,0] = Y_1_0
	system.compute(times)
	return 1.0 + planet.lightcurve

def plot_model(times, data, fpfs, label='', ax = None):

	if ax is None:
		fig = gcf()
		ax = fig.add_subplot(111)

	ax.plot(times, data, label=label) # Green Curve

	# Universal
	ax.set_ylim(1-fpfs,1 + 2.5*fpfs)

	ax.set_xlabel('Time [day]', fontsize=30)
	ax.set_ylabel('normalized flux', fontsize=30)

	ax.set_title('STARRY:Compare Influence of FpFs vs CosAmp', fontsize=30)

	ax.legend(loc=0, fontsize=20)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(20)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(20)

	return ax

planet_info = exomast_api.exoMAST_API(planet_name='HD 189733 b')

phase = np.linspace(0, 1.0, 1000)
times = phase*planet_info.orbital_period + planet_info.transit_time

fpfs = 3000/1e6
Y_1_0 = 0.5
lmax = 1
lambda0 = 90

data = generate_starry_model(times, planet_info, fpfs, lmax = lmax, 
								lambda0 = lambda0, Y_1_0 = Y_1_0)

nrg_ratio = 0.0
temp_night = 200.
delta_T = 700.
T_star = 4645.

n_layers = 5
brightness_model="zhang"

spider_params = sp.ModelParams(brightness_model=brightness_model)
spider_params.n_layers = n_layers

n_pts = 100
# times = planet_info.transit_time
# times = times + np.linspace(0, planet_info.orbital_period, n_pts)

init_model = generate_spiderman_model(times, planet_info, nrg_ratio, 
						temp_night, delta_T, T_star, spider_params)

data_err = 1.0

args = {'data':data, 
		'data_err':data_err, 
		'times':times, 
		'planet_info':planet_info, 
		'spider_params':spider_params, 
		'T_star':T_star}

params = [nrg_ratio, temp_night, delta_T]

def chisq_sp(params, args):
	if any(np.array(params) < 0): return np.inf

	data = args['data']
	data_err = args['data_err']
	times = args['times']
	planet_info = args['planet_info']
	spider_params = args['spider_params']
	T_star = args['T_star']

	nrg_ratio, temp_night, delta_T = params

	model = generate_spiderman_model(times, planet_info, nrg_ratio, temp_night,
								delta_T, T_star, spider_params)

	return np.sum(((data-model)/data_err)**2)

def chisq_lmfit(model_params, args):
	data = args['data']
	data_err = args['data_err']
	times = args['times']
	planet_info = args['planet_info']
	spider_params = args['spider_params']
	T_star = args['T_star']

	nrg_ratio = model_params['nrg_ratio']
	temp_night = model_params['temp_night']
	delta_T = model_params['delta_T']

	model = generate_spiderman_model(times, planet_info, nrg_ratio, temp_night,
								delta_T, T_star, spider_params)

	return ((data-model)/data_err)**2

from lmfit import Parameters, Minimizer
from multiprocessing import cpu_count

res = op.minimize(chisq_sp, params, args=(args), method='powell')

nrg_ratio_fit, temp_night_fit, delta_T_fit = res.x

fit_model = generate_spiderman_model(times, planet_info, nrg_ratio_fit, 
					temp_night_fit, delta_T_fit, T_star, spider_params)

print('Initializing Parameters')
initialParams = Parameters()
initialParams.add_many(
	('nrg_ratio', np.max([nrg_ratio_fit,0.0]), True, 0.0, 1.0),
	('temp_night', temp_night_fit, True, 0.0, np.inf),
	('delta_T', delta_T_fit, True, 0.0, np.inf))

from functools import partial
partial_residuals  = partial(chisq_lmfit, args = args)

mle0  = Minimizer(partial_residuals, initialParams, nan_policy = 'omit')
fitResult = mle0.leastsq(initialParams)

def logprior_func(p):
	for key, val in p.items():
		# Uniform Prior
		if val.min >= val.value >= val.max: return -np.inf
	
	# Establish that the limb darkening parameters 
	#  cannot sum to 1 or greater
	#  Kipping et al 201? and Espinoza et al 201?
	if 'u1' in p.keys() and 'u2' in p.keys():
		if p['u1'] + p['u2'] >= 1: return -np.inf
	
	return 0

def lnprob(p):
	logprior = logprior_func(p)
	if not np.isfinite(logprior):
		return -np.inf
	
	resid = partial_residuals(p)
	
	s = p['err_mod']
	resid *= 1 / s
	resid *= resid
	resid += np.log(2 * np.pi * s**2)
	
	return -0.5 * np.sum(resid) + logprior

mle0.params.add('err_mod', value=1, min=1e-4, max=10)

mini  = Minimizer(lnprob, mle0.params, nan_policy='omit')

n_steps = 1000
n_walkers = 100
n_burn = int(n_steps*0.2)
n_thin = 10
n_temps = 1
starting_values = np.array([val.value for val in initialParams.values() if val.vary])

pos = np.random.normal(starting_values, 1e-2*abs(starting_values), (n_walkers, len(starting_values)))

reuse_sampler = False
n_workers = cpu_count()#-1
is_weighted = True
seed = None# 42

start = time()

print('MCMC routine in progress...') 
mcmc_fit = mini.emcee(params=mle0.params, steps=n_steps, nwalkers=n_walkers, 
					burn=n_burn, thin=n_thin, ntemps=n_temps,
					pos=pos, reuse_sampler=reuse_sampler, workers=n_workers,
					is_weighted=is_weighted, seed=seed)

nrg_ratio_mcmc, temp_night_mcmc, delta_T_mcmc = res.x
nrg_ratio_mcmc, temp_night_mcmc, delta_T_mcmc, err_mod_mcmc = np.median(
												mcmc_fit.flatchain, axis=0)

mcmc_model = generate_spiderman_model(times, planet_info, nrg_ratio, 
						temp_night, delta_T, T_star, spider_params)

plot_now = True
if plot_now: 
	ax = plot_model(times, data, fpfs, label='STARRY', ax=None)
	ax = plot_model(times, init_model, fpfs, label='Init Model', ax=ax)
	ax = plot_model(times, fit_model, fpfs, label='MLE Model', ax=ax)
	ax = plot_model(times, mcmc_model, fpfs, label='MCMC Model', ax=ax)

corner_plot = False
if corner_plot:
	report_errors(mcmc_fit.params)
	
	res = mcmc_fit
	res_var_names = np.array(res.var_names)
	res_flatchain = np.array(res.flatchain)
	res_df = DataFrame(res_flatchain.copy(), columns=res_var_names)
	res_df = DataFrame(res_flatchain.copy(), columns=res_var_names)
	res_df.sort_index('columns', inplace=True)

	def add_newline(label): return label + '\n'

	color = 'indigo'
	
	n_sigma = 3
	levels = [0.682689492137, 0.954499736104, 0.997300203937, 0.999936657516, 0.999999426697, 0.999999998027]
	
	title_kwargs = dict(columnwise = True)
	
	corner_kwargs = {}
	corner_kwargs['color'] = color
	corner_kwargs['labels'] = list(map(add_newline,res_df.columns))
	corner_kwargs['plot_datapoints'] = False
	corner_kwargs['bins'] = 50
	corner_kwargs['plot_density'] = False
	corner_kwargs['smooth'] = True
	corner_kwargs['fill_contours'] = True
	corner_kwargs['levels'] = levels[:n_sigma]
	corner_kwargs['show_titles'] = True
	corner_kwargs['title_fmt'] = '0.2e'
	corner_kwargs['columnwise_titles'] = True
	corner_kwargs['title_kwargs'] = dict(fontsize=20)

	corner.corner(res_df, **corner_kwargs)

	plt.show()