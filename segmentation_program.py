import pylab as plt
import numpy as np
import imageio
import os
import logging
import glob
from scipy.signal import convolve2d
from skimage.transform import  rescale, resize
from skimage import measure
from skimage.filters import  median, threshold_yen , threshold_multiotsu
from scipy import ndimage
from draw_radial_line import draw_radial_lines
from define_border import define_border,distanza
logging.basicConfig(level=logging.INFO)


'''
This function pre-process the image.
'''
def process_img(image, smooth_factor, scale_factor):

    k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
    im_conv = convolve2d(image, k )
    image_normalized = rescale(im_conv, 1/scale_factor)/np.max(im_conv)
    im_log = 255*np.log10(image+1)
    im_log = im_log.astype('uint8')
    im_median = median(im_log, np.ones((5,5)))
    im_res = resize(im_median, (126,126))
    im_log_normalized = im_res/np.max(im_res)

    return im_log_normalized, image_normalized


def find_center(x_max, y_max):
    
    '''if the point with maximum intensity is too far away from the ROI center,
    the center is chosen as the center of the rectangle. This is also the starting point of rays.'''
    
    if((np.abs(x_max-(x2-x1)/2)<(4/5)*(x1+(x2-x1)/2)) or (np.abs(y_max-(y2-y1)/2)<(4/5)*(y1+(y2-y1)/2))):
        x_center = x1+int((x2-x1)/2)
        y_center = y1+int((y2-y1)/2)
    else:
        x_center = x_max[0]
        y_center = y_max[0]
    center = [x_center, y_center]
    return center


def 