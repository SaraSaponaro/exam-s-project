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
    J = generic_filter(normalized, np.std, size=3)
    #plt. figure()
    #plt.imshow(J, cmpa='gray')

    B_points=[]
    shape=np.shape(normalized)
    roughborder=np.zeros(shape)

    for it in range(1, NL):
        #itero sulla terza dimensione: Ray_masks immagini 162x162, 32 (raggi)
        Jmasked=J*Ray_masks[:,:,it]     #J*raggi=maschera dell'img
        threshold = np.max()
        Jmasked=Jmasked*imbinarize(ROI)
        [c1, c2]= np.where(Jmasked==np.max(Jmasked))    #mi devo accertare che prenda gli indici giusti
        '''devo riguardare su matlab
        B_points=[B_points; c1(1), c2(1) ,J(c1(1),c2(1))]
        '''
        roughborder[c1(1),c2(2)]=normalized[c1(1),c2(1)]  #copio pixel img all'interno della matrice


    #come posso chiudere i bordi ?!
