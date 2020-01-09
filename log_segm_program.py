import pylab as plt
import numpy as np
import imageio
import os
import logging
import glob
import time
from scipy.signal import convolve2d
from skimage.transform import  rescale, resize
from skimage import measure
from skimage.filters import  median, threshold_yen , threshold_multiotsu
from skimage import exposure
from PIL import Image
from scipy import ndimage
from draw_radial_line import draw_radial_lines
from define_border import define_border_min, distanza, define_border_max
logging.basicConfig(level=logging.INFO)



def process_img(image, smooth_factor, scale_factor):
    '''
    C=np.min(image)/np.max(image)
    #if C == 0:
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    '''
    k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
    im_conv=convolve2d(image, k )       #per ridurre il rumore (alte frequenze)
    image_n = rescale(im_conv, 1/scale_factor)/np.max(im_conv)
    
    '''try logaritmic trasformation and averange filter'''
    im_log=255*np.log10(image+1)
    im_log=im_log.astype('uint8')
    im_median= median(im_log, np.ones((5,5)))
    im_res = resize(im_median, (126,126))
    im_log_n = im_res/np.max(im_res)
    
    plt.figure()
    plt.title('image')
    plt.imshow(image_n)
    plt.grid(True)
    plt.colorbar()
    plt.show()

    return im_log_n, im_res, image_n


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
    #read all files
    logging.info('Enter images path.')
    logging.info('Luigi -> /Users/luigimasturzo/Documents/esercizi_fis_med/large_sample/*.png')
    logging.info('Sara -> /Users/sarasaponaro/Desktop/exam_cmpda/large_sample/*.png')

    file_path=str(input('file path = : '))

    fileID = glob.glob(file_path)


    for _ in range (3,4):
        f = open('center_list.txt', 'a')
        image=imageio.imread(fileID[_])


        filename, file_extension = os.path.splitext(fileID[_])
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
             
        im_log_n, im_resized , image_n= process_img(image, smooth_factor, scale_factor)

        logging.info('Starting the image segmentation.')

        decision  = 'si'

        while (decision != 'no'):

            logging.info('Enter ROIs coordinates that contains the mass.')
            y1=int(input('Enter y1: '))
            x1=int(input('Enter x1: '))
            y2=int(input('Enter y2: '))
            x2=int(input('Enter x2: '))


            ROI=np.zeros(np.shape(image_n))                 #ones
            ROI[y1:y2,x1:x2]=image_n[y1:y2,x1:x2]
            y_max,x_max=np.where(ROI==np.max(ROI))
            if (len(x_max) == 1):
                center = find_center(x_max, y_max)
            else:
                center = find_center(x_max[0], y_max[0])


            logging.info('Showing ROI and center.' )
            plt.figure()
            plt.title('ROI')
            plt.imshow(image_n)
            plt.imshow(ROI, alpha=0.3)
            plt.plot(center[0], center[1], 'r.')
            plt.colorbar()
            plt.show()

            print('Do you want to change your coordinates?')
            decision=input('Answer yes or no: ')

        f.write('{} \t {} \t {}\n'.format(filename, center[0], center[1]))
        
        logging.info('Finding the border.')
        
        img=np.zeros(np.shape(image_n))
        val=threshold_yen(image_n)
        img[image_n >= val] = 1
        img[image_n < val] = 0
        
        plt.figure('first step -> Yen threshold')
        plt.imshow(img*ROI)
        plt.plot(center[0], center[1], 'r.')
        plt.show()
        
        logging.info('finding multiple otsu threshold')
        p=np.zeros((y2-y1,x2-x1))
        pp=np.zeros(np.shape(image_n))
        p=img[y1:y2,x1:x2]*ROI[y1:y2,x1:x2]
        val1, val2=threshold_multiotsu(p)
        p[p>val2]=0
        p[p<val1]=0
        pp[y1:y2,x1:x2]=p
        
        fill=np.zeros(np.shape(image_n))
        contours = measure.find_contours(pp, 0)
        arr=  contours[0]
        arr = arr.flatten('F')
        arr=arr.astype('int')
        y = arr[0:(int(len(arr)/2))]
        x = arr[(int(len(arr)/2)):]
        fill[y,x]=1
        fill=ndimage.binary_fill_holes(fill).astype(int)
        
        plt.figure('second step -> mask after double th.')
        plt.imshow(fill)
        plt.show()
        
        #define length of ray.
        R=int(distanza(x1,y1,x2,y2)/2)
        roi=np.zeros(np.shape(image_n))
        roi[y1:y2,x1:x2]=1
  
        logging.info('Refining segmentation results.')
        roughborder_r=np.zeros(np.shape(im_log_n))
        __px=[]
        __py=[]
        xx_r=[]
        yy_r=[]
        for _ in range(0,len(x)):
            center_raff = [x[_], y[_]]
            Ray_masks_raff = draw_radial_lines(img*ROI,center_raff,int(R/R_scale),NL)
            roughborder_raff, _y , _x ,_px, _py= define_border_max(img*ROI, NL, fill ,size_nhood_variance, Ray_masks_raff)
            roughborder_r+=roughborder_raff
            xx_r=np.hstack((xx_r,_x))
            yy_r=np.hstack((yy_r,_y))
            __px+=_px
            __py+=_py
            
        xx_r=xx_r.astype('int')
        yy_r=yy_r.astype('int')
            
        
        fill_raff=ndimage.binary_fill_holes(roughborder_r).astype(int)
        #fill_raff[__py,__px]=0
        #fill_raff[yy_r,xx_r]=0
        

        plt.figure()
        plt.title('Final mask of segmented mass.')
        plt.imshow(fill_raff)
        plt.show()

        logging.info('Showing result')
        mass_only = fill_raff*image_n

        plt.figure()
        plt.title('Segmented mass.')
        plt.imshow(mass_only)
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title('confronto.')
        conf=imageio.imread('/Users/sarasaponaro/Desktop/exam_cmpda/large_sample_Im_segmented_ref/'+str(filename)+'_mass_mask.png')
        plt.imshow(conf)
        plt.imshow(image_n, alpha=0.3)
        plt.imshow(mass_only, alpha=0.7)
        plt.show()


        '''
        logging.info('Savening result.')
        im = im_resized.astype(np.uint8)
        im = Image.fromarray(im, mode='P')
        im.save(file_out)

        fill_raff = fill_raff.astype(np.int8)
        im1 = Image.fromarray(fill, mode='L')
        im1.save(mask_out)
        '''
        f.close()
