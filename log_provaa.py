import pylab as plt
import numpy as np
import imageio
import os
import logging
import glob
import time
from scipy.signal import convolve2d
from skimage.draw import line_nd
from skimage.transform import  rescale
from skimage.filters import  median
from PIL import Image
from scipy import ndimage
from draw_radial_line import draw_radial_lines
from define_border import define_border_new, distanza, define_border_final
logging.basicConfig(level=logging.INFO)

def process_img(image, smooth_factor, scale_factor):
    
    '''try logaritmic trasformation an averange filter'''
    im_log=255*np.log10(image+1)
    im_log=im_log.astype('uint8')
    im_median= median(im_log, np.ones((5,5)))
    im_res = rescale(im_median, 1/scale_factor)
    im_norm = im_res/np.max(im_res)
    
    
    k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
    im_conv=convolve2d(image, k )       #per ridurre il rumore (alte frequenze)
    image_modify = rescale(image, 1/scale_factor)/np.max(image)

    plt.figure()
    plt.title('log image')
    plt.imshow(image_modify)
    plt.grid(True)
    plt.colorbar()
    plt.show()

    return im_norm, im_res, image_modify


def find_center(x_max, y_max):
    '''if the point with maximum intensity is too far away from the ROI center,
    the center is chosen as the center of the rectangle. This is also the starting point of rays.'''
    if((np.abs(x_max-(x2-x1)/2)<(4/5)*(x1+(x2-x1)/2)) or (np.abs(y_max-(y2-y1)/2)<(4/5)*(y1+(y2-y1)/2))):
        x_center=x1+int((x2-x1)/2)
        y_center=y1+int((y2-y1)/2)
    else:
        x_center=x_max[0]
        y_center=y_max[0]
    center=[x_center, y_center]
    return (center)

if __name__ == '__main__':
    logging.info('Reading files')
    #fileID='0016p1_2_1.png'
    #fileID='0025p1_4_1.png'
    fileID='0036p1_1_1.png'

    #read all files
    fileID = glob.glob('img/*.png')

    for _ in range (0, 1):
        f = open('center_list.txt', 'a')
        image=imageio.imread(fileID[0])


        filename, file_extension = os.path.splitext(fileID[0])
        filename=os.path.basename(filename)
        path_out = 'result/'

        if (os.path.exists(path_out)==False):
            logging.info ('Creating folder for save results.\n')
            os.makedirs('result')

        file_out=path_out+filename+'_resized'+file_extension
        mask_out=path_out+filename+'_mask'+file_extension

        logging.info('Defining parameters ')
        smooth_factor= 8
        scale_factor= 8
        size_nhood_variance=5
        NL=33
        R_scale=5

        logging.info('Processing image {}'.format(filename))
        im_norm, im_resized , image_modify= process_img(image, smooth_factor, scale_factor)

        logging.info('Starting the image segmentation.')

        decision  = 'si'

        while (decision != 'no'):

            logging.info('Enter ROIs coordinates that contains the mass.')
            y1=int(input('Enter y1: '))
            x1=int(input('Enter x1: '))
            y2=int(input('Enter y2: '))
            x2=int(input('Enter x2: '))


            ROI=np.ones(np.shape(im_norm))
            ROI[y1:y2,x1:x2]=im_norm[y1:y2,x1:x2]
            y_max,x_max=np.where(ROI==np.min(ROI))
            center = find_center(x_max, y_max)


            logging.info('Showing ROI and center.' )
            plt.figure()
            plt.title('ROI')
            plt.imshow(im_norm)
            plt.imshow(ROI, alpha=0.3)
            plt.plot(center[0], center[1], 'r.')
            plt.colorbar()
            plt.show()

            print('Do you want to change your coordinates?')
            decision=input('Answer yes or no: ')

        #f.write('{} \t {} \t {}\n'.format(filename, center[0], center[1]))


        logging.info('Drawing radial lines.')
        #define length of ray.
        R=int(distanza(x1,y1,x2,y2)/2)

        Ray_masks=draw_radial_lines(ROI,center,R,NL)

        plt.figure()
        plt.title('Some of rays found.')
        plt.imshow(Ray_masks[0])      
        plt.imshow(image_modify, alpha=0.3)
        #plt.imshow(ROI, alpha=0.5)
        plt.plot(center[0],center[1], 'r.')
        plt.colorbar()
        plt.show()

        logging.info('Finding the border.')
        roi=np.zeros(np.shape(im_norm))
        roi[y1:y2,x1:x2]=1

        roughborder,bordofinale_x,bordofinale_y=define_border_final(image_modify, NL, roi,size_nhood_variance, Ray_masks)
        fill=ndimage.binary_fill_holes(roughborder).astype(int)

        plt.figure()
        plt.title('Mask of segmented mass.')
        plt.imshow(fill)
        plt.show()

        #FINO A QUI TUTTO OK
        
        logging.info('Refining segmentation results.')
        p_x=[]
        p_y=[]
        d=[]
        rr_arr=np.array([0])
        cc_arr=np.array([0])
        p=np.zeros(np.shape(im_norm))
        
        mask=image_modify*fill
        
        matrix=np.zeros(np.shape(im_norm))
        for _ in range (0, len(bordofinale_x)):
            raggi=np.zeros(np.shape(im_norm))
            coords = line_nd((bordofinale_x[_],bordofinale_y[_]),(center[0],center[1]))
            raggi[coords[1][0:2],coords[0][0:2]]=1
            matrix =mask * raggi
            #matrix[matrix==0]=10
            w = np.where(matrix==np.max(matrix))
            p+=matrix
            p_y.append(w[0][0])
            p_x.append(w[1][0])
            d.append(matrix[w[0][0],w[1][0]])
        roughborder_n=np.zeros(np.shape(im_norm)) 
        for _ in range(0,len(bordofinale_x)-1):
            coords = line_nd((p_y[_],p_x[_]),(p_y[_+1],p_x[_+1]))
            roughborder_n[coords[1],coords[0]]=1
            rr_arr=np.hstack((rr_arr,coords[0]))
            cc_arr=np.hstack((cc_arr,coords[1]))
            
        fill_raff=ndimage.binary_fill_holes(roughborder_n).astype(int)
        
        plt.figure()
        plt.title('prova')
        plt.imshow(fill_raff)
       

        conf=imageio.imread('ref/0016p1_2_1_mass_mask.png')
        plt.figure()
        plt.imshow(conf)
        plt.imshow(fill_raff*image_modify, alpha=0.6)
        plt.imshow(fill*image_modify, alpha=0.6)
        plt.show()
        
        
        
        