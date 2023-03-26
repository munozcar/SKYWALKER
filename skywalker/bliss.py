from scipy import spatial
from pylab import *
from . import utils


def nearestIndices(xcenters, ycenters, knot_tree, neighbors=4):
    """
        Args:
        xcenters (array): array of x coordinates of each center.
        ycenters (array): array of y coordinates of each center.
        knot_tree (spatial.KDTree): spatial.KDTree(knots)
        Returns:
        array: array of arrays, each with the indices of the 4 nearest knots to each element in x/y-centers.
        """
    return array([
        utils.nearest(xc, yc, neighbors, knot_tree)
        for xc, yc in zip(xcenters, ycenters)
    ])


def createGrid(xcenters, ycenters, x_bin_size, y_bin_size):
    """
    :param point_list:  array of lists with (x,y) coordinates of each center.
    :param x_bin_size: x length of each rectangle in the knot grid.
    :param y_bin_size: y length of each rectangle in the knot grid.
    :return: array of lists with (x,y) coordinates of each vertex in knot grid.
    """
    xmin, xmax = min(xcenters), max(xcenters)
    ymin, ymax = min(ycenters), max(ycenters)
    return [
        (x, y)
        for x in arange(xmin, xmax, x_bin_size)
        for y in arange(ymin, ymax, y_bin_size)
    ]


def associate_fluxes(knots, near_indices, xcenters, ycenters, fluxes):
    """
    Args:
        :param knots: array of lists with (x,y) coordinates of each vertex in the knot grid.
        :param near_indices: array of arrays, each with the indices of the 4 nearest knots to each element in y/x-centers.
        :param xcenters: array of lists with x coordinates of each center.
        :param ycenters: array of lists with y coordinates of each center.
        :param fluxes: array of fluxes corresponding to each element in x/y-centers.
    :return:
        Array with the mean flux associated with each knot of the grid.
    """
    # knot_fluxes = [[] for k in knots]
    knot_fluxes = [[]] * len(knots)

    for kp in range(len(xcenters)):
        N = near_indices[kp][0]
        knot_fluxes[N].append(fluxes[kp])

    return [mean(flux) if len(flux) != 0 else 0 for flux in knot_fluxes]


def generate_delta_x_delta_y(xcenters, ycenters, knots, near_indices):
    """
        :param xcenters: array of lists with x coordinates of each center.
        :param ycenters: array of lists with y coordinates of each center.
        :param knots: array of lists with (x,y) coordinates of each vertex in the knot grid.
        :param near_indices: array of arrays, each with the indices of the 4 nearest knots to each element in y/x-centers.
    """
    delta_x1 = zeros(len(xcenters))
    delta_y1 = zeros(len(ycenters))
    for kp, (xc, yc) in enumerate(zip(xcenters, ycenters)):
        delta_x1[kp] = abs(xc - knots[near_indices[kp][0]][0])
        delta_y1[kp] = abs(yc - knots[near_indices[kp][0]][1])

    return delta_x1, delta_y1


def interpolateFlux(
        knots, knot_fluxes, delta_x1, delta_y1, near_indices, x_bin_size,
        y_bin_size, norm_factor):
    """
        Args:
        knots (array): array of lists with (x,y) coordinates of each vertex in the knot grid.
        knot_fluxes (array): array of the flux associated with each knot.
        delta_x1 (array): array with delta x-coordinates of each center to knot1_X.
        deltay1 (array): array with delta y-coordinates of each center to knot1_Y.
        near_indices (array): array of arrays, each with the indices of the 4 nearest knots to each element in x/y-centers.
        x_bin_size (float): x length of each rectangle in the knot grid.
        y_bin_size (float): y length of each rectangle in the knot grid.
        norm_factor (float): (1/x_bin_size) * (1/y_bin_size)
        Returns:
        array: array of interpolated flux at each point in x/y-centers.
    """
    interpolated_fluxes = np.zeros(len(delta_x1))
    for kp, (dx1, dy1) in enumerate(zip(delta_x1, delta_y1)):
        nearest_fluxes = [knot_fluxes[i] for i in near_indices[kp]]
        # If any knot has no flux, use nearest neighbor interpolation.
        if 0 in nearest_fluxes:
            N = near_indices[kp][0]
            interpolated_fluxes[kp] = knot_fluxes[N]

        # Else, do bilinear interpolation
        else:
            # Normalize distances with factor
            # Interpolate
            dx2 = x_bin_size - dx1
            dy2 = x_bin_size - dy1

            interpolated_fluxes[kp] = norm_factor * (
                dx1 * dy2 * nearest_fluxes[0]
                + dx2 * dy2 * nearest_fluxes[1]
                + dx2 * dy1 * nearest_fluxes[2]
                + dx1 * dy1 * nearest_fluxes[3]
            )
    return interpolated_fluxes


def BLISS(
        xcenters, ycenters, fluxes, knots, near_indices, x_bin_size=0.01,
        y_bin_size=0.01, norm_factor=10000):
    """
        Args:
        xcenters (array): array of x-coordinates of each center.
        ycenters (array): array of y-coordinates of each center.
        fluxes (array): array of fluxes corresponding to each element in x/y-centers.
        knots (array): array of lists with (x,y) coordinates of each vertex in the knot grid.
        near_indices (array): array of arrays, each with the indices of the 4 nearest knots to each element in x/y-centers.
        x_bin_size (float): x length of each rectangle in the knot grid.
        y_bin_size (float): y length of each rectangle in the knot grid.
        norm_factor (float): (1/x_bin_size) * (1/y_bin_size)
        Returns:
        array: array of interpolated flux at each point in x/y-centers.
        """
    mean_knot_fluxes = associate_fluxes(
        knots,
        near_indices,
        xcenters,
        ycenters,
        fluxes
    )

    delta_x1, delta_y1 = generate_delta_x_delta_y(
        xcenters,
        ycenters,
        knots,
        near_indices
    )

    return interpolateFlux(
        knots=knots,
        knot_fluxes=mean_knot_fluxes,
        delta_x1=delta_x1,
        delta_y1=delta_y1,
        near_indices=near_indices,
        x_bin_size=x_bin_size,
        y_bin_size=y_bin_size,
        norm_factor=norm_factor
    )
