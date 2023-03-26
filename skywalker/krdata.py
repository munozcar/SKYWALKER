import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
from tqdm import tqdm

# Global constants.
y, x = 0, 1
ppm = 1e6
day_to_seconds = 86400
zero = 0.0


def find_qhull_one_point(point, x0, y0, np0, inds, kdtree=None):
    dx = x0[inds[point]] - x0[point]
    dy = y0[inds[point]] - y0[point]

    if np0.sum() != 0.0:
        dnp = np0[inds[point]] - np0[point]

    sigx = np.std(dx)
    sigy = np.std(dy)

    x_exp = -dx**2./(2.0*sigx**2.)
    y_exp = -dy**2. / (2.*sigy**2.)
    exponent = x_exp + y_exp
    if dnp.sum() != 0.0:
        signp = np.std(dnp)
        np_exp = -dnp**2./(2.*signp**2.)

        exponent = exponent + np_exp

    gw_temp = np.exp(exponent)
    gw_temp_sum = gw_temp.sum()
    return np.zeros(len(gw_temp)) if gw_temp_sum == 0 else gw_temp / gw_temp_sum


def gaussian_weights_and_nearest_neighbors(
        xpos, ypos, npix=None, inds=None, n_nbr=50, returnInds=False,
        a=1.0, b=0.7, c=1.0, expansion=1000., n_cores=1):
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012, Fraine etal 2013

        Taken from N. Lewis IDL code:

            Construct a 3D surface (or 2D if only using x and y) from the data
            using the qhull.pro routine.  Save the connectivity information for
            each data point that was used to construct the Delaunay triangles (DT)
            that form the grid.  The connectivity information allows us to only
            deal with a sub set of data points in determining nearest neighbors
            that are either directly connected to the point of interest or
            connected through a neighboring point

        Python Version:
            J. Fraine    first edition, direct translation from IDL 12.05.12
    '''
    n, k = inds.shape  # This is the number of nearest neighbors you want
    func = partial(find_qhull_one_point, x0=xpos, y0=ypos, np0=npix, inds=inds)

    if n_cores > 1:
        raise Exception(
            'Check to make sure that Multiprocessing is working correctly '
            '-- examine the Activity Monitor.'
        )
        pool = Pool(n_cores)

        gw_list = pool.starmap(func, zip(range(n)))

        pool.close()
        pool.join()
    else:
        gw_list = []
        for idx in range(n):
            gw_list.append(func(idx))

    if returnInds:
        return np.array(gw_list), inds
    else:
        return np.array(gw_list)


def sensitivity_map(residuals, ind_kdtree, gw_kdtree):
    return np.sum(residuals[ind_kdtree] * gw_kdtree, axis=1)
