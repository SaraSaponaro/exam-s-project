'''programma per crare il bordo grezzo della lesione;
il bordo è dato dai punti dei raggi con la massimo std'''
import pylab as plt
import numpy as np
from scipy.ndimage.filters import generic_filter

'''funsione che binarizza img'''
def imbinarize(img, threshold):
    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img




def max_var_point(normalized, ROI, Ray_masks, NL, nhood):
    '''standard deviation filter (3x3) for img normalized'''
    J = generic_filter(normalized, np.std, size=3)
    #plt. figure()
    #plt.imshow(J, cmpa='gray')

    B_pints=[]
    shape=np.shaoe(normalized)
    roughborder=np.zeros(shape)

    for it in range(1, NL):
        Jmasked=J*Ray_masks[:,:,it]     #itero sulla terza dimensione
        '''riguarda come è fatta ROI e decidi come impostare la soglia'''
        Jmasked=Jmasked*imbinarize(ROI, threshold)

        [c1, c2]= np.where(Jmasked==np.max(Jmasked[:]);
        '''devo riguardare su matlab
        B_points=[B_points; c1(1), c2(1) ,J(c1(1),c2(1))]
        '''
        roughborder[c1(1),c2(2)]=normalized[c1(1),c2(1)]

    #come posso chiudere i bordi ?!
