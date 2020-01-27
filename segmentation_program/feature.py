import imageio
import logging
import glob
import os
import pylab as plt
import numpy as np
from skimage import measure
from define_border import distanza
from scipy.stats import  kurtosis, skew
from skimage.morphology import convex_hull_image
from skimage.measure import EllipseModel
from skimage.transform import resize
logging.basicConfig(level=logging.INFO)


def linear(x1, y1, x2, y2):
    """
    Find angular coefficient and intercept of a straight line given two points.

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
    m,q : float
        angular coefficient and slope of the computed line.
    """
    m = (y2-y1)/(x2-x1)
    q = (y1 - x1*(y2-y1))/(x2-x1)
    return m, q

def mass_area(mask_only):
    """
    Finds the number of pixels inside the mass

    Parameters
    ----------
    mask_only : numpy.ndarray
        This is a 2D array representing the mask of the segmented image.
        The mass is filled with 1, the background is filled with 0.

    Returns
    ----------
    area : int
        the area (number of pixels) of the segmented mass.

    """
    a = np.where(mask_only != 0)
    area = np.shape(a)[1]
    return area

def mass_perimeter(mask_only):
    """
    Summing up the number of pixels on the boundary's mass

    Parameters
    ----------
    mask_only : numpy.ndarray
        This is a 2D array representing the mask of the segmented image.
        The mass is filled with 1, the background is filled with 0.

    Returns
    ----------
    perimeter : int
        the perimeter (number of pixels on the boundary) of the segmented mass.

    """
    contours = measure.find_contours(mask_only, 0, fully_connected='high')
    return len(contours[0])

def circularity(area, perimetro):
    c = 4*np.pi*area/(perimetro**2)
    return c

def NRL(mask_only, center_x, center_y, perimetro):
    """
    Finds the border of the mask and computes the distance between the boundary pixels and the center.
    It also calculates the normalized distance, the mean and the standard deviation of the normalized distance.

    Parameters
    ----------
    mask_only : numpy.ndarray
        This is a 2D array representing the mask of the segmented image.
        The mass is filled with 1, the background is filled with 0.
    center_x : int
        This number indicates the abscissa of the center of the mass.
    center_y : int
        This number indicates the ordinate of the center of the mass.
    perimetro : int
        This number stands for the perimeter of the segmented mass.

    Returns
    ----------
    d : float
        array containing distances between boundary pixels and the center.
    d_mean : float
        mean value of d.
    d_norm : float
        normalized d.
    std : float
        standard deviation of d.
    """
    contours = measure.find_contours(mask_only, 0, fully_connected='high')
    arr = contours[0]
    arr = arr.flatten('F')
    y = arr[0:(int(len(arr)/2))]
    x = arr[(int(len(arr)/2)):]
    d = distanza(center_x,center_y, x, y)
    d_norm = d/(np.max(d))
    d_mean = np.sum(d_norm)/perimetro
    s = np.sum((d_norm-d_mean)**2)
    std = np.sqrt(s/perimetro)
    return d, d_mean, d_norm, std

def Radial_lenght_entropy(d_norm):
    """
    Find a probabilistic measure computed from the histogram of the normalized radial lenght.

    Parameters
    ----------
    d_norm : numpy.ndarray
        This is a 1D array containing normalized distances between boundary pixels and the center.

    Returns
    ----------
    E : float
        Radial Length Entropy.
    """
    __, bins, _ = plt.hist(d_norm, 5, density=1)
    E=0
    for i in range(0, 5):
        mask1 = d_norm < bins[i+1]
        mask2 = d_norm > bins[i]
        p = len(d_norm[np.logical_and(mask1, mask2)])/len(d_norm)
        E += - p*np.log(p + 2.2e-16)
    return E

def cross_zero(d, d_mean):

    """
    Counts the number of times that the radial distance from the center to boundary pixels overcomes the mean distance.

    Parameters
    ----------
    d : numpy.ndarray
        This is a 1D array containing distances between boundary pixels and the center.

    d_mean : float
        Mean of d.

    Returns
    ----------
    c : int
        number of times that the radial distance from the center to boundary pixels overcomes the mean distance.
    """
    c = np.where(d>=np.mean(d))
    return len(c[0])

def axis(mask_only):
    """
    Finds minimum and maximum distance connecting two boundary pixels passing trough the center.

    Parameters
    ----------
    mask_only : numpy.ndarray
        This is a 2D array representing the mask of the segmented image.
        The mass is filled with 1, the background is filled with 0.

    Returns
    ----------
    max_axis : float
        maximum distance of the line passing between boundary pixels and the center.
    min_axis : float
        minimum distance of the line passing between boundary pixels and the center.
    """
    contours = measure.find_contours(mask_only, 0, fully_connected='high')
    arr = contours[0].flatten('F')
    y = arr[0:int(len(arr)/2)]
    x = arr[int(len(arr)/2):]
    z = np.hstack((x, y)).reshape(2,-1).T
    ell = EllipseModel()
    ell.estimate(z)
    _, __, a, b, ___ = ell.params
    axes = [a, b]
    return 2*np.min(axes), 2*np.max(axes)

def var_ratio(d):
    """
    Finds the maximum variation of distance from the mean. It computes the mean and the standard deviation of the dominant variations.

    Parameters
    ----------
    d : numpy.ndarray
        This is a 1D array containing distances between boundary pixels and the center.

    Returns
    ----------
    mean, std : float
        mean value and standard deviation of variations of distance from the mean value.

    """
    vm = np.max(d-(np.mean(d)))/2
    mean = np.mean(np.abs(d-(np.mean(d)))>=vm)
    std = np.std(np.abs(d-(np.mean(d))>=vm))
    return mean, std

def convexity(mass,area):
    """
    Finds the smallest convex containing the mass. It returns the ratio between the mass area and the area of convex hull.

    Parameters
    ----------
    mass : numpy.ndarray
        This is a 2D array representing the mask of the segmented image.
        The mass is filled with its original value, the background is filled with 0.

    Returns
    ----------
    ratio : float
        ratio between the mass area and the area of the smallest convex hull.

    """
    hull = convex_hull_image(mass)
    a = np.where(hull != 0)
    area_hull = np.shape(a)[1]
    ratio=area/area_hull
    return ratio

def mass_intensity(mass):

    """
    Finds the mean and the standard deviation of the grey level intensity value of image.

    Parameters
    ----------
    mass : numpy.ndarray
        This is a matrix where the segmented mass is filled with its original value and the background is set to 0.

    Returns
    ----------
    mean, std : float
        mean value and standard deviation of pixels' intensity of the segmented mass.
    """
    mass = mass/np.max(mass)
    mean = np.mean(mass)
    std = np.std(mass)
    return mean, std


if __name__ == '__main__':
    logging.info('Reading files.')
    files = glob.glob('large_sample_Im_segmented_ref/*_resized.png')
    masks = glob.glob('result/*_mask.png')
    files.sort()
    masks.sort()

    f_ref = open('../txt/feature_alg.txt', 'w')
    fML = open ('../txt/feature_alg_ML.txt', 'w')
    #f.write('filename \t classe \t area \t perimeter \t circularity \t mu_NRL \t std_NRL \t zero_crossing \t max_axis \t min_axis \t')
    #f.write('mu_VR \t std_VR \t RLE \t convexity \t mu_I \t std_I \t kurtosis \t skewness\n')
    for index, item in enumerate(masks):
        filename, file_extension = os.path.splitext(item)
        filename = os.path.basename(filename)
        filename = filename[:-5]
        "1: malignant 2:benign"
        classe = filename[-1]

        mask_only = imageio.imread(item)
        img = imageio.imread(files[index])
        mask_only = np.delete(mask_only, 0, 0)
        mask_only = np.delete(mask_only, 0, 1)
        mass = img*mask_only

        a=np.where(mass!=0)
        center_intensity=np.where(mass==np.max(mass))

        x1=np.min(a[1])
        y1=np.min(a[0])
        x2=np.max(a[1])
        y2=np.max(a[0])

        center_x = x1+int((x2-x1)/2)
        center_y = y1+int((y2-y1)/2)

        area = mass_area(mask_only)
        perimeter = mass_perimeter(mask_only)
        circ = circularity(area,perimeter)
        d, mu_NRL, d_norm, std_NRL = NRL(mask_only, center_x, center_y, perimeter)
        cross0 = cross_zero(d, mu_NRL)
        rmin,rmax  = axis(mask_only)
        vm, vs = var_ratio(d)
        E = Radial_lenght_entropy(d_norm)
        conv = convexity(mass, area)
        im, istd = mass_intensity(mass)
        intensity = np.reshape(mass[mass!=0], -1)
        kurt = kurtosis(intensity, fisher = False)
        sk = skew(intensity)
        '''
        reference:
        f_ref.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t '.format(filename, classe, area, perimeter, circ ,mu_NRL, std_NRL, cross0))
        f_ref.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}   \n'.format(rmax, rmin, vm, vs, E, conv, im, istd, kurt, sk))
        fML.write('{} \t{} \t{} \t{} \t{} \t{} \t'.format(filename, classe, area, circ ,mu_NRL, std_NRL))
        fML.write('{} \t{} \t{} \t{} \n'.format( E, istd, kurt, sk))
        '''
        f_ref.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t '.format(filename, classe, area, perimeter, circ ,mu_NRL, std_NRL, cross0))
        f_ref.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}\n'.format(rmax, rmin, vm, vs, E, conv, im, istd, kurt, sk))
        fML.write('{} \t{} \t{} \t{} \t{} \t{} \t'.format(filename, classe, area, circ ,mu_NRL, std_NRL))
        fML.write('{} \t{}\t{} \t{}\t{}\t{} \n'.format(vm, E, conv, istd, kurt, sk))

    fML.close()
    f_ref.close()
