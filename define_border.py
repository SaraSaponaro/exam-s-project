from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu
import numpy as np
import math
import pylab as plt
from skimage.draw import line_nd
'''distanaza euclidea'''
def distanza(x1,y1,x2,y2):
    distanza_euclidea=np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distanza_euclidea

'''funsione che binarizza img'''
def imbinarize(img):
    thresh = threshold_otsu(img)
    img[img >= thresh] = 1
    img[img < thresh] = 0
    return img

def define_border_min(im_norm, NL, ROI,size_nhood_variance, Ray_masks):
    roughborder=np.zeros(np.shape(im_norm))            #np.zeros(np.shape(im_norm))

    p_x=[]
    p_y=[]
    d=[]

    rr_arr=np.array([0])
    cc_arr=np.array([0])
    for _ in range (0, NL):
        Jmasked=im_norm*Ray_masks[_]     #J*raggi=maschera dell'img
        Jmasked=Jmasked*ROI
        Jmasked[Jmasked==0]=10
        w = np.where(Jmasked==np.min(Jmasked))
        p_y.append(w[0][0])
        p_x.append(w[1][0])
        d.append(Jmasked[w[0][0],w[1][0]])

    for _ in range(0,NL-1):
        coords = line_nd((p_y[_],p_x[_]),(p_y[_+1],p_x[_+1]))
        roughborder[coords[0],coords[1]]=1
        rr_arr=np.hstack((rr_arr,coords[0]))
        cc_arr=np.hstack((cc_arr,coords[1]))

    return roughborder, rr_arr, cc_arr


def define_border_max(im_norm, NL, ROI,size_nhood_variance, Ray_masks):
    roughborder=np.zeros(np.shape(im_norm))            #np.zeros(np.shape(im_norm))
    J=generic_filter(im_norm, np.std, size=size_nhood_variance)
    '''
    plt.figure('std')
    plt.imshow(J)
    plt.show()
    '''
    p_x=[]
    p_y=[]
    d=[]
    rr_arr=np.array([0])
    cc_arr=np.array([0])
    for _ in range (0, NL):
        Jmasked=J*Ray_masks[_]     #J*raggi=maschera dell'img
        Jmasked=Jmasked*ROI
        w = np.where(Jmasked==np.max(Jmasked))
        p_y.append(w[0][0])
        p_x.append(w[1][0])
        d.append(Jmasked[w[0][0],w[1][0]])

    for _ in range(0,NL-1):
        coords = line_nd((p_y[_],p_x[_]),(p_y[_+1],p_x[_+1]))
        roughborder[coords[0],coords[1]]=1
        rr_arr=np.hstack((rr_arr,coords[0]))
        cc_arr=np.hstack((cc_arr,coords[1]))

    return roughborder, rr_arr, cc_arr
