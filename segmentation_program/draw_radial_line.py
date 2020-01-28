import math
import numpy as np

def pol2cart(rho, theta):
    """
    Function that computes a trasformation from polar to cartesian coordinates.

    Parameters
    ----------
    rho : float
        distance between the point and the center in the plane.
    theta : float
        angle formed between the ray and the positive half-line of the abscissa.

    Returns
    ----------
    x : float
        abscissa of the point.
    y : float
        ordinate of the point.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

def draw_radial_lines(ROI, center, R, NL):
    """
    Creates a binary mask of size ROI, where NL radial lines of lenght R are depicted starting from the center.

    Parameters
    ----------
    ROI : numpy.ndarray
        2d array containing the region of interest selected.
    center : list
        list containing the ordinata and abscissa of the center.
    R : int
        lenght of the ray used in the algorithm.
    NL :
        number of rays starting from one point.
    """
    theta = np.linspace(0, 2*np.pi, NL)
    Ray_masks = []
    rho = np.arange(R)

    for ii in range(0, NL):
        xx = []
        yy = []
        for jj in range(0, R):
            x, y = pol2cart(rho[jj], theta[ii])
            xx.append(center[0]+math.ceil(x))
            yy.append(center[1]+math.ceil(y))
        line1 = np.column_stack((xx, yy))
        Ray_mask = np.zeros(np.shape(ROI))

        for __, item in enumerate(line1):
            i = item[0]
            j = item[1]
            Ray_mask[j, i] = 1

        Ray_masks.append(Ray_mask)

    return np.asarray(Ray_masks)
