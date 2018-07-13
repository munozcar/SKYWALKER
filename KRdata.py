from functools         import partial

from multiprocessing   import Pool, cpu_count
from numpy             import loadtxt, array, sqrt, median, sum, zeros, int64, transpose, float64, std, exp, isfinite, arange, sin

from scipy.spatial     import cKDTree

from tqdm              import tqdm

def find_qhull_one_point(point, x0, y0, np0, inds, kdtree=None):
    dx  = x0[inds[point]] - x0[point]
    dy  = y0[inds[point]] - y0[point]

    if np0.sum() != 0.0:
        dnp         = np0[inds[point]] - np0[point]

    sigx  = std(dx )
    sigy  = std(dy )

    if dnp.sum() != 0.0:
        signp     = std(dnp)
        exponent  = -dx**2./(2.0*sigx**2.) + -dy**2./(2.*sigy**2.) + -dnp**2./(2.*signp**2.)
    else:
        exponent  = -dx**2./(2.0*sigx**2.) + -dy**2./(2.*sigy**2.)

    gw_temp = exp(exponent)

    return gw_temp / gw_temp.sum()

def gaussian_weights_and_nearest_neighbors(xpos, ypos, npix = None, inds = None, n_nbr = 50, returnInds=False,
                      a = 1.0, b = 0.7, c = 1.0, expansion = 1000., nCores=1):
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
    #The surface fitting performs better if the data is scattered about zero
    x0  = (xpos - median(xpos))/a
    # x0 = x0/std(x0)
    y0  = (ypos - median(ypos))/b
    # y0 = y0/std(y0)
#
    if npix is not None and bool(c):
        np0 = sqrt(npix)
        np0 = (np0 - median(np0))/c
        features  = transpose((y0, x0, np0))
    else:
        features  = transpose((y0, x0))

        if sum(np0) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')

    if inds is None:
        kdtree    = cKDTree(features * expansion) #Multiplying `features` by 1000.0 avoids precision problems
        inds      = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]

        print('WARNING: Because `inds` was not provided, we must now compute and return it here')
        returnInds= True

    n, k   = inds.shape                           # This is the number of nearest neighbors you want

    func  = partial(find_qhull_one_point, x0=x0, y0=y0, np0=np0, inds=inds)

    if nCores > 1:
        raise Exception('Check to make sure that Multiprocessing is working correctly -- examine the Activity Monitor.')
        pool  = Pool(nCores)

        gw_list = pool.starmap(func, zip(range(n)))

        pool.close()
        pool.join()
    else:
        gw_list = []
        for idx in range(n):
            gw_list.append(func(idx))

    if returnInds:
        return array(gw_list), inds
    else:
        return array(gw_list)

if __name__ == '__main__':
    nPts = int(1e5)

    import numpy as np
    import matplotlib.pyplot as plt

    xpos = 0.35*sin(np.arange(0,nPts) / 1500 + 0.5) + 15 + np.random.normal(0,0.2,nPts)
    ypos = 0.35*sin(np.arange(0,nPts) / 2000 + 0.7) + 15 + np.random.normal(0,0.2,nPts)
    npix = 0.25*sin(np.arange(0,nPts) / 2500 + 0.4) + 15 + np.random.normal(0,0.2,nPts)
    flux = 1+0.01*(xpos - xpos.mean()) + 0.01*(ypos - ypos.mean()) + 0.01*(npix - npix.mean())

    n_nbr   = 50
    points  = transpose([xpos,ypos,npix])
    kdtree  = cKDTree(points)

    ind_kdtree  = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:] # skip the first one because it's the current point

    # `gaussian_weights_and_nearest_neighbors` only returns the gaussian weights in the indices are provided
    gw_kdtree   = gaussian_weights_and_nearest_neighbors(xpos   , ypos   , npix   , ind_kdtree  )
    gkr_kdtree  = sum(flux[ind_kdtree]  * gw_kdtree, axis=1)

    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(flux        , '.', ms=1, alpha=0.5)
    ax1.plot(gkr_kdtree  , '.', ms=1, alpha=0.5)

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(flux - gkr_kdtree  , '.', ms=1, alpha=0.5)

    ax2.set_title('Scipy.cKDTree Gaussian Kernel Regression')

    ax2.set_ylim(-0.0005,0.0005)
    plt.show()
