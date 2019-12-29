from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu
import numpy as np
import math

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

def define_border(im_norm, NL, ROI,size_nhood_variance, Ray_masks):
    J = generic_filter(im_norm, np.std, size=size_nhood_variance)

    roughborder=np.zeros(np.shape(im_norm))

    p_x=[]
    p_y=[]
    d=[]

    bordofinale_x=[]
    bordofinale_y=[]

    for _ in range (0, NL):
        Jmasked=J*Ray_masks[_]     #J*raggi=maschera dell'img
        Jmasked=Jmasked*imbinarize(ROI)
        w = np.where(Jmasked==np.max(Jmasked))
        p_y.append(w[0][0])
        p_x.append(w[1][0])
        d.append(Jmasked[w[0][0],w[1][0]])

    "riempio il bordo tra due elementi adiacenti di p_x,p_y"
    def find_border(_, p_x, p_y):

        'liste vuote che riempirò con i pixel trovati'
        bordo_x=[]
        bordo_y=[]

        distanza_finale=100
        while (distanza_finale >= 1):     #finchè non raggiungo il pixel stop

            'trovo i pixel vicini'
            vicino_x=[p_x[_]-1,p_x[_]-1,p_x[_],p_x[_]+1,p_x[_]+1,p_x[_]+1,p_x[_]+1,p_x[_]-1]
            vicino_y=[p_y[_],p_y[_]+1,p_y[_]+1,p_y[_]+1,p_y[_],p_y[_]-1,p_y[_]-1,p_y[_]-1]

            'trovo la distanza e scelgo quella minima'
            distanza_list=[]
            c=0
            for __ in range(0,len(vicino_x)):
                c=distanza(vicino_x[__],vicino_y[__],p_x[_+1], p_y[_+1])
                distanza_list.append(c)

            distanza_list=np.asarray(distanza_list)
            'scelgo il pixel a cui mi corrisponde la distanza minima rispetto a quello di stop'
            d=np.argmin(distanza_list)                 #np.where(distanza_list==distanza_list.min())
            distanza_finale=distanza_list[int(d)]

            'coordinate del pixel prescelto'
            pixel_x=vicino_x[int(d)]
            pixel_y=vicino_y[int(d)]

            'mi salvo la posizione del pixel, perchè alla fine dovrò concatenare tutte queste liste ed otterrò il bordo'
            bordo_x.append(pixel_x)
            bordo_y.append(pixel_y)

            'il mio nuovo pixel da cui trovare tutti i vicini è quello prescelto'
            p_x[_]=pixel_x
            p_y[_]=pixel_y
        return bordo_x, bordo_y

    for _ in range(0,NL-1):
        bordoo_x=[]
        bordoo_y=[]
        bordoo_x, bordoo_y = find_border(_, p_x, p_y)
        roughborder[bordoo_y, bordoo_x]=1
        bordofinale_x += bordoo_x
        bordofinale_y += bordoo_y

    return roughborder, bordofinale_x,bordofinale_y
