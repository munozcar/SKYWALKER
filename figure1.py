import utils as utils
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fluxes, times, flux_errs, npix, pld_coeffs, xcenters, ycenters, xwidth_keys, ywidth_keys = utils.extractData('../GJ1214b_TSO/data/GJ1214b_group0_ExoplanetTSO_transit1.joblib.save')

keepinds = utils.removeOutliers(xcenters=xcenters, ycenters=ycenters)

phase = ((times[keepinds] - 54966.524918) % 1.58040481) / 1.58040481
phase[phase > 0.5] -= 1.0

fluxes[keepinds] = fluxes[keepinds]/np.median(fluxes[keepinds])
rem = fluxes[keepinds] < 1.02


ax = plt.subplot2grid((2,2),(0,1), rowspan=2)
plt.scatter(xcenters[keepinds][rem], ycenters[keepinds][rem], c=fluxes[keepinds][rem], s=2, marker='o', cmap='jet')
plt.xlabel('x-centroid', fontsize=16)
plt.ylabel('y-centroid', fontsize=16)
plt.colorbar(orientation="vertical")
ax2 = plt.subplot2grid((2,2),(0,0))
plt.scatter(phase[rem], xcenters[keepinds][rem], c=fluxes[keepinds][rem], s=1, marker='o', cmap='jet')
plt.ylabel('x-centroid', fontsize=16)

ax3 = plt.subplot2grid((2,2),(1,0))
plt.scatter(phase[rem], ycenters[keepinds][rem], c=fluxes[keepinds][rem], s=1, marker='o', cmap='jet')
plt.xlabel('Phase', fontsize=16)
plt.ylabel('y-centroid', fontsize=16)
plt.show()
