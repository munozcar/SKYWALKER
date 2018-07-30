from sklearn.externals import joblib
import utils as utils
from scipy import spatial
import numpy as np

flux_key='phots'
time_key='times'
flux_err_key='noise'
eff_width_key = 'npix'
pld_coeff_key = 'pld'
ycenter_key='ycenters'
xcenter_key='xcenters'
ywidth_key='ywidths'
xwidth_key='xwidths'

def cat_transits(filearray):
    cat_fluxes = []
    cat_times = []
    cat_flux_errs = []
    cat_npix = []
    cat_pld_intensities = [[],[],[],[],[],[],[],[],[]]
    cat_xcenters = []
    cat_ycenters = []
    cat_xwidths = []
    cat_ywidths = []

    for file in filearray:
        fluxes, times, flux_errs, npix, pld_intensities, xcenters, ycenters, xwidths, ywidths = utils.extractData(file=file)

        cat_fluxes = cat_fluxes + list(fluxes)
        cat_times = cat_times + list(times)
        cat_flux_errs = cat_flux_errs + list(flux_errs)
        cat_npix = cat_npix + list(npix)
        i = 0
        for pix in pld_intensities:
            cat_pld_intensities[i] = cat_pld_intensities[i] + list(pix)
            i += 1
        cat_xcenters = cat_xcenters + list(xcenters)
        cat_ycenters = cat_ycenters + list(ycenters)
        cat_xwidths = cat_xwidths + list(xwidths)
        cat_ywidths = cat_ywidths + list(ywidths)


    cat_fluxes = np.array(cat_fluxes)
    cat_times = np.array(cat_times)
    cat_flux_errs = np.array(cat_flux_errs)
    cat_npix = np.array(cat_npix)
    cat_pld_intensities = np.array([np.array(pix) for pix in cat_pld_intensities])
    cat_xcenters = np.array(cat_xcenters)
    cat_ycenters = np.array(cat_ycenters)
    cat_xwidths = np.array(cat_xwidths)
    cat_ywidths = np.array(cat_ywidths)

    current_transit = {}
    current_transit[time_key] = cat_times
    current_transit[flux_key] = cat_fluxes
    current_transit[flux_err_key] = cat_flux_errs
    current_transit[eff_width_key] = cat_npix
    current_transit[pld_coeff_key] = cat_pld_intensities
    current_transit[xcenter_key] = cat_xcenters
    current_transit[ycenter_key] = cat_ycenters
    current_transit[xwidth_key] = cat_xwidths
    current_transit[ywidth_key] = cat_ywidths

    joblib.dump(current_transit, 'cat_all.joblib.save')


filearray = ['../GJ1214b_TSO/data/GJ1214b_group0_ExoplanetTSO.joblib.save',
'../GJ1214b_TSO/data/GJ1214b_group1_ExoplanetTSO.joblib.save',
 '../GJ1214b_TSO/data/GJ1214b_group2_ExoplanetTSO.joblib.save',
'../GJ1214b_TSO/data/GJ1214b_group3_ExoplanetTSO.joblib.save',
'../GJ1214b_TSO/data/GJ1214b_group4_ExoplanetTSO.joblib.save',
'../GJ1214b_TSO/data/GJ1214b_group5_ExoplanetTSO.joblib.save',
'../GJ1214b_TSO/data/GJ1214b_group6_ExoplanetTSO.joblib.save']

cat_transits(filearray)
