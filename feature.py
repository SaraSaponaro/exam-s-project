import pylab as plt
import numpy as np
import imageio
import logging
import glob
import os
from skimage import measure
from define_border import distanza
from scipy.stats import  kurtosis, skew
from skimage.morphology import convex_hull_image
from skimage.measure import EllipseModel



def linear(x1,y1,x2,y2):
    """
    Find angular coefficient and intercept of a straight line given two points
    """
    m = (y2-y1)/(x2-x1)
    q = (y1 - x1*(y2-y1))/(x2-x1)
    return m,q

def mass_area(mask_only):
    """
    Finds the number of pixels inside the mass
    """
    a = np.where(mask_only != 0)
    area = np.shape(a)[1]
    return area

def mass_perimeter(mask_only):
    """
    Summing up the number of pixels on the boundary's mass
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
    """
    n, bins, _ = plt.hist(d_norm, 5, density=1)
    E=0
    for i in range(0, 5):
        mask1 = d_norm < bins[i+1]
        mask2 = d_norm > bins[i]
        p = len(d_norm[np.logical_and(mask1,mask2)])/len(d_norm)
        E += - p*np.log(p + 2.2e-16)
    return E

def cross_zero(d,d_mean):
   """
   Counts the number of times that the radial distance from the center to boundary pixels overcomes the mean distance.
   """
   c = np.where(d>=np.mean(d))
   return len(c[0])


def axis(mask_only):
    """
    Finds minimum and maximum distance connecting two boundary pixels passing trough the center.
    """
    contours = measure.find_contours(mask_only, 0, fully_connected='high')
    arr = contours[0].flatten('F')
    y = arr[0:int(len(arr)/2)]
    x = arr[int(len(arr)/2):]
    z = np.hstack((x,y)).reshape(2,-1).T
    ell = EllipseModel()
    ell.estimate(z)
    xc, yc, a, b, theta = ell.params
    axes = [a, b]
    return np.min(axes), np.max(axes)

def var_ratio(d):
    """
    Finds the maximum variation of distance from the mean. It computes the mean and the standard deviation of the dominant variations.
    """
    vm = np.max(d-(np.mean(d)))/2
    mean = np.mean(np.abs(d-(np.mean(d)))>=vm)
    std = np.std(np.abs(d-(np.mean(d))>=vm))
    return mean, std

def convexity(mass,area):
    """
    Finds the smallest convex containing the mass. It returns the ratio between the mass area and the area of convex hull.
    """
    hull = convex_hull_image(mass)
    a = np.where(hull != 0)
    area_hull = np.shape(a)[1]
    return area/area_hull
    
def mass_intensity(mass):
    """
    Finds the mean and the standard deviation of the grey level intensity value of image.
    """
    mass = mass/np.max(mass)
    mean = np.mean(mass)
    std = np.std(mass)
    return mean,std


if __name__ == '__main__':
    logging.info('Reading files.')
    files = glob.glob('/Users/sarasaponaro/Desktop/exam_cmpda/large_sample_Im_segmented_ref/*_resized.png')
    masks = glob.glob('/Users/sarasaponaro/Desktop/exam_cmpda/large_sample_Im_segmented_ref/*_mask.png')
    #files = glob.glob('/Users/luigimasturzo/Documents/esercizi_fis_med/large_sample_Im_segmented_ref/*_resized.png')
    #masks = glob.glob('/Users/luigimasturzo/Documents/esercizi_fis_med/large_sample_Im_segmented_ref/*_mask.png')
    files.sort()
    masks.sort()
    nametxt = 'feature_ref.txt'
    nametxt_m = 'feature_m.txt'
    nametxt_b= 'feature_b.txt'
    f = open(nametxt, 'w')
    fm = open(nametxt_m, 'w')
    fb = open(nametxt_b, 'w')
    #f.write('filename \t classe \t area \t perimeter \t circularity \t mu_NRL \t std_NRL \t zero_crossing \t max_axis \t min_axis \t')
    #f.write('mu_VR \t std_VR \t RLE \t convexity \t mu_I \t std_I \t kurtosis \t skewness\n')
    for index, item in enumerate(masks):
        filename, file_extension = os.path.splitext(item)
        filename = os.path.basename(filename)
        filename = filename[:-10]
        "1: malignant 2:benign"
        classe = filename[-1]

        mask_only = imageio.imread(item)
        img = imageio.imread(files[index])
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
        p = mass_perimeter(mask_only)
        circ = circularity(area,p)
        d, mu_NRL, d_norm, std_NRL = NRL(mask_only, center_x, center_y, p)
        cross0 = cross_zero(d, mu_NRL)
        rmin,rmax  = axis(mask_only)
        vm, vs = var_ratio(d)
        E = Radial_lenght_entropy(d_norm)
        conv = convexity(mass, area)
        im, istd = mass_intensity(mass)
        intensity = np.reshape(mass[mass!=0], -1)
        kurt = kurtosis(intensity, fisher = False)
        sk = skew(intensity)
        if classe == '1':

            fm.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t '.format(filename, classe, area, p, circ ,mu_NRL, std_NRL, cross0))
            fm.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}   \n'.format(rmax, rmin, vm, vs, E, conv, im, istd, kurt, sk))
        elif classe == '2':

            fb.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t '.format(filename, classe, area, p, circ ,mu_NRL, std_NRL, cross0))
            fb.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}   \n'.format(rmax, rmin, vm, vs, E, conv, im, istd, kurt, sk))
        f.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t '.format(filename, classe, area, p, circ ,mu_NRL, std_NRL, cross0))
        f.write('{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}   \n'.format(rmax, rmin, vm, vs, E, conv, im, istd, kurt, sk))
    fm.close()
    fb.close()
    f.close()
