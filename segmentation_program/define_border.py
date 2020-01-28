import numpy as np
from scipy.ndimage.filters import generic_filter
from skimage.draw import line_nd

def distanza(x1, y1, x2, y2):
    """Function that returns the euclidean distance.

    Parameters
    ----------
    x1 : int
        the abscissa of the first point.
    y1 : int
        the ordinate of the first point.
    x2 : int
        the abscissa of the second point.
    y2 : int
        the abscissa of the second point.

    Returns
    ----------
    distanza_euclidea : float
        distance between two points.

    """

    distanza_euclidea = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distanza_euclidea

def define_border(img, NL, ROI, size_nhood_variance, Ray_masks):
    '''
    The function creates a roughborder of the mass constituted by the points on the radial lines with maximum standard deviation.

    Parameters
    ----------
    img : numpy.ndarray
        image to be analized.
    NL : int
        number of rays starting from one point.
    ROI : numpy.ndarray
        2d array containing the region of interest selected.
    size_nhood_varaince : int
        dimension of the kernel used in standard deviation filter.
    Ray_masks : numpy.ndarray
        3D array wherre the third dimension rapresents a single ray.

    Returns
    ----------
    roughborder : numpy.ndarray
        binary matrix where 1 stands for the line connecting the new boundary points

    '''
    x_max = []
    y_max = []
    roughborder = np.zeros(np.shape(img))
    J = generic_filter(img, np.std, size=size_nhood_variance)

    for i in range(0, NL):
        Jmasked = J*Ray_masks[i]
        Jmasked = Jmasked*ROI
        if np.max(Jmasked) != 0:
            w = np.where(Jmasked == np.max(Jmasked))
            y_max.append(w[0][0])
            x_max.append(w[1][0])

    for j in range(0, len(x_max)-1):
        coords = line_nd((y_max[j], x_max[j]), (y_max[j+1], x_max[j+1]))
        roughborder[coords[0], coords[1]] = 1

    return roughborder
