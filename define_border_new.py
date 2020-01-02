from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu
import numpy as np
import math
import pylab as plt
from skimage.draw import line, line_aa, line_nd


'''funsione che binarizza img'''
def imbinarize(img):
    thresh = threshold_otsu(img)
    img[img >= thresh] = 1
    img[img < thresh] = 0
    return img

def define_border_new(im_norm, NL, ROI,size_nhood_variance, Ray_masks):
    J = generic_filter(im_norm, np.std, size=size_nhood_variance)
    roughborder=np.zeros(np.shape(im_norm))

    p_x=[]
    p_y=[]
    d=[]


    rr_arr=np.array([0])
    cc_arr=np.array([0])
    

    for _ in range (0, NL):
        Jmasked=J*Ray_masks[_]     #J*raggi=maschera dell'img
        Jmasked=Jmasked*imbinarize(ROI)
        w = np.where(Jmasked==np.max(Jmasked))
        p_y.append(w[0][0])
        p_x.append(w[1][0])
        d.append(Jmasked[w[0][0],w[1][0]])

    
    for _ in range(0,NL-1):
        #rr,cc,__ = line_aa(p_x[_],p_y[_],p_x[_+1],p_y[_+1])
        #roughborder[cc,rr]=1
        #bordofinale_x += rr
        #bordofinale_y += cc
        #rr_arr=np.hstack((rr_arr,rr))
        #cc_arr=np.hstack((cc_arr,cc))


        coords = line_nd((p_x[_],p_y[_]),(p_x[_+1],p_y[_+1]))
        roughborder[coords[1],coords[0]]=1
        '''print('---------',_,'----------')
        print(coords)
        print(coords[0])
        print(rr)
        print('-----------')
        print(type(coords))
        print('-----------')
        print(np.shape(coords))'''

        
        rr_arr=np.hstack((rr_arr,coords[0]))
        cc_arr=np.hstack((cc_arr,coords[1]))
        




    return roughborder, rr_arr, cc_arr
