import argparse
import glob
import os
import logging
import pylab as plt
import numpy as np
import imageio
from PIL import Image
from scipy import signal, ndimage
from skimage.transform import rescale
from skimage import measure
from skimage.morphology import label
from skimage.filters import hessian
from draw_radial_line import draw_radial_lines
from define_border import define_border, distanza
logging.basicConfig(level=logging.INFO)

_description = 'Computer-aided diagnosis (CAD) system for characterising masses.'

def process_img(image, smooth_factor, scale_factor):
    """
    This function pre-process the image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be processed.
    smooth_factor : int
        The number that caracterizes the smoothing.
    scale_factor :  int
        The number that indicates how much you want to rescale the image.

    Returns
    ----------
    image_normalized : numpy.ndarray
        A rescaled and smoothed image of the input image.
    """
    k = np.ones((smooth_factor, smooth_factor))/smooth_factor**2
    im_conv = signal.convolve2d(image, k)
    image_normalized = rescale(im_conv, 1/scale_factor)
    return image_normalized

def find_center(x_max, y_max, y1, x1, y2, x2):
    """
    If the point with maximum intensity is too far away from the ROI center,
    the center is choosen as the center of the rectangle. This is also the starting point of rays.

    Parameters
    ----------

    y_max : int
        This number indicates the ordinate of the maximum intensity pixel.
    x_max : int
        This number indicates the abscissa of the maximum intensity pixel.
    y_1 : int
        This number indicates the ordinate of the upper left corner of the ROI.
    x_1 : int
        This number indicates the abscissa of the upper left corner of the ROI.
    y_2 : int
        This number indicates the ordinate of the lower right corner of the ROI.
    x_2 : int
        This number indicates the abscissa of the lower right corner of the ROI.

    Returns
    ----------
    center : list
        a list containing the ordinate and abscissa of the ROI center.

    """
    if((np.abs(x_max-(x2-x1)/2) < (4/5)*(x1+(x2-x1)/2)) or (np.abs(y_max-(y2-y1)/2) < (4/5)*(y1+(y2-y1)/2))):
        x_center = x1+int((x2-x1)/2)
        y_center = y1+int((y2-y1)/2)
    else:
        x_center = x_max
        y_center = y_max
    center = [x_center, y_center]
    return center

def segmentation():
    """
    This function performs the real segmentation of the input image.

    Parameters
    ----------

    file_path : string
        The path where the images are stored.
    """
    logging.info('Reading files')
    fileID = glob.glob('large_sample/*.png')
    for item in range(171,177):
        f = open('center_list.txt', 'a')
        image = imageio.imread(fileID[item])
        filename, file_extension = os.path.splitext(fileID[item])
        filename = os.path.basename(filename)
        path_out = 'result/'

        if os.path.exists(path_out) is False:
            logging.info('Creating folder for save results.\n')
            os.makedirs('result')
        mask_out = path_out + filename + '_mask' + file_extension

        logging.info('Defining parameters ')
        smooth_factor = 8
        scale_factor = 8
        size_nhood_variance = 5
        NL = 33
        R_scale = 5

        logging.info('Processing image {}'.format(filename))
        image_n = process_img(image, smooth_factor, scale_factor)
        conf = imageio.imread('large_sample_Im_segmented_ref/'+str(filename)+'_mass_mask.png')
        plt.figure()
        plt.subplot(121)
        plt.title('image {}'.format(filename))
        plt.imshow(image_n)
        plt.imshow(conf, alpha=0.1)
        plt.grid()
        plt.subplot(122)
        plt.imshow(image_n)
        plt.grid()
        plt.show()

        logging.info('Starting the image segmentation.')
        decision = 'si'

        while decision != 'no':
            logging.info('Enter ROIs coordinates that contains the mass.')
            y1 = int(input('Enter y1: '))
            x1 = int(input('Enter x1: '))
            y2 = int(input('Enter y2: '))
            x2 = int(input('Enter x2: '))

            ROI = np.zeros(np.shape(image_n))
            img_hessian = hessian(image_n)
            ROI[y1:y2, x1:x2] = img_hessian[y1:y2, x1:x2]
            y_max, x_max = np.where(ROI == np.max(ROI))

            if len(x_max) == 1:
                center = find_center(x_max, y_max, y1, x1, y2, x2)
            else:
                center = find_center(x_max[0], y_max[0], y1, x1, y2, x2)

            logging.info('Showing ROI and center.')
            plt.figure()
            plt.title('ROI')
            plt.imshow(image_n)
            plt.imshow(ROI, alpha=0.3)
            plt.plot(center[0], center[1], 'r.')
            plt.colorbar()
            plt.show()

            if args.show != None:
                plt.figure()
                plt.title('equalize')
                plt.imshow(img_hessian*ROI)
                plt.plot(center[0], center[1], 'r.')
                plt.colorbar()
                plt.show()
            print('Do you want to change your coordinates?')
            decision = input('Answer yes or no:')

        f.write('{} \t {} \t {}\t {}\t {}\t {}\t {}\n'.format(filename, center[0], center[1], y1, x1, y2, x2))

        logging.info('Cleaning non-conneted regions.')
        ROI[ROI < 1] = 0
        fill = ndimage.binary_fill_holes(ROI).astype(int)
        fill, n = label(fill, return_num=True)

        count = []
        for i in range(1, n+1):
            count.append(np.count_nonzero(fill == i))
        w = np.where(count == np.max(count))
        fill[fill != w[0]+1] = 0

        if args.show is not None:
            plt.figure()
            plt.title('cleaning image.')
            plt.imshow(fill)
            plt.show()

        logging.info('Find the coutours.')
        contours = measure.find_contours(fill, 0, fully_connected='high')
        arr = contours[0].flatten('F').astype('int')
        y = arr[0:(int(len(arr)/2))]
        x = arr[(int(len(arr)/2)):]

        logging.info('Final refinement: ')
        R = int(distanza(x1, y1, x2, y2)/2)
        roughborder = np.zeros(np.shape(image_n))

        for i in range(len(x)):
            print(len(x)-i)
            center_raff = [x[i], y[i]]
            Ray_masks_raff = draw_radial_lines(ROI, center_raff, int(R/R_scale), NL)
            roughborder_raff = define_border(image_n, NL, fill, size_nhood_variance, Ray_masks_raff)
            roughborder += roughborder_raff

        fill_raff = ndimage.binary_fill_holes(roughborder).astype(int)

        frequency_raff = np.reshape(fill_raff, -1)
        frequency = np.reshape(fill, -1)

        if list(frequency_raff).count(0) > list(frequency).count(0):
            print('change')
            fill_raff = fill

        if args.show != None:
            plt.figure()
            plt.title('Final mask of segmented mass.')
            plt.imshow(fill_raff)
            plt.show()

        plt.figure()
        plt.title('confronto.')
        plt.subplot(1, 2, 1)
        conf = imageio.imread('large_sample_Im_segmented_ref/'+str(filename)+'_mass_mask.png')
        plt.imshow(conf)
        plt.imshow(fill_raff, alpha=0.7)
        plt.subplot(1, 2, 2)
        plt.imshow(fill_raff)
        plt.show()

        fill_raff = fill_raff.astype(np.int8)
        im1 = Image.fromarray(fill_raff, mode='L')
        im1.save(mask_out)

        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-s', '--show', help='Do you want to show the images of process?')
    args = parser.parse_args()
    segmentation()
