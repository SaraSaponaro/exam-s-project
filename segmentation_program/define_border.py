import numpy as np
from scipy.ndimage.filters import generic_filter
from skimage.draw import line_nd

'''Function that returns the euclidean distance.'''
def distanza(x1,y1,x2,y2):
    distanza_euclidea = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distanza_euclidea


'''
The function creates a roughborder of the mass constituted by the points on the radial lines with maximum standard deviation.
'''
def define_border(img, NL, ROI,size_nhood_variance, Ray_masks):
    
    x_max=[]
    y_max=[]
    roughborder = np.zeros(np.shape(img))           
    J = generic_filter(img, np.std, size=size_nhood_variance)
    
    for _ in range (0, NL):
        Jmasked = J*Ray_masks[_]
        Jmasked = Jmasked*ROI
        if(np.max(Jmasked)!=0):
            w = np.where(Jmasked==np.max(Jmasked))
            y_max.append(w[0][0])
            x_max.append(w[1][0])
        
    for _ in range(0,len(x_max)-1):
        coords = line_nd((y_max[_],x_max[_]),(y_max[_+1],x_max[_+1]))
        roughborder[coords[0],coords[1]]=1
        
    return roughborder
