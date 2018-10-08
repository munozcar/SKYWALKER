from sklearn.externals import joblib

import corner.corner
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

res = joblib.load('emcee.joblib.save')

res_var_names = np.array(res.var_names)
res_flatchain = np.array(res.flatchain)
res_df = DataFrame(res_flatchain, columns=res_var_names)
res_df = res_df.drop(['u2','slope0'], axis=1)

sigmaTc = np.std(res_df['deltaTc'])
meanTc = np.mean(res_df['deltaTc'])

sigmatd = np.std(res_df['tdepth'])
meantd = np.mean(res_df['tdepth'])

sigmau1 = np.std(res_df['u1'])
meanu1 = np.mean(res_df['u1'])

sigmau1 = np.std(res_df['u1'])
meanu1 = np.mean(res_df['u1'])

sigmaint = np.std(res_df['intcept0'])
meanint = np.mean(res_df['intcept0'])

sigmaf = np.std(res_df['f'])
meanf = np.mean(res_df['f'])

corner.corner(res_df, color='darkred', plot_datapoints=False, bins=50, plot_density=False, smooth=True, fill_contours = True, levels=[0.68, 0.95, 0.997], show_titles=True, title_fmt = '0.5e', range = [(meanTc-(6*sigmaTc), meanTc+(6*sigmaTc)), (meantd-(6*sigmatd), meantd+(6*sigmatd)), (meanu1-(6*sigmau1), meanu1+(6*sigmau1)), (meanint-(6*sigmaint), meanint+(6*sigmaint)), (meanf-(6*sigmaf), meanf+(6*sigmaf))])
plt.show()
