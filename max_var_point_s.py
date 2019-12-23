'''programma per crare il bordo grezzo della lesione;
il bordo è dato dai punti dei raggi con la massimo std'''
import pylab as plt
import numpy as np
from scipy.ndimage.filters import generic_filter
from skimage.filters import threshold_otsu

'''funsione che binarizza img'''
def imbinarize(img):
    thresh = threshold_otsu(img)
    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img


def max_var_point(normalized, ROI, Ray_masks, NL, nhood):
    '''standard deviation filter (3x3) for img normalized'''
    J = generic_filter(normalized, np.std, size=nhood)
    #plt. figure()
    #plt.imshow(J, cmpa='gray')

    B_points=[]
    roughborder=np.zeros(np.shape(normalized))

    Jmasked=J*Ray_masks     #J*raggi=maschera dell'img
    Jmasked=Jmasked*imbinarize(ROI)
    w = np.where(Jmasked==np.max(Jmasked))    #mi devo accertare che prenda gli indici giusti
    '''prova a far girare tuttoil programam per essere sicura che funxioni bene '''
    list=[w(0), w(1) , J[w(0),w(1)]]
    B_points.extend(list)
    roughborder[w(0),w(1)]=normalized[w(0),w(1)]  #copio pixel img all'interno della matrice


    #come posso chiudere i bordi ?!
