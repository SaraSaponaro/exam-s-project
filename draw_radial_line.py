import numpy as np
import math

'''
Function that computes a trasformation from polar to cartesian coordinates.
'''
def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

'''
Creates a binary mask of size ROI, where NL radial lines of lenght R are depicted starting from the center.
'''

def draw_radial_lines(ROI,center,R,NL):

    theta = np.linspace(0,2*np.pi,NL)
    Ray_masks = []
    rho = np.arange(R)

    for _ in range(0,NL):
        xx = []
        yy = []  
        
        for __ in range (0, R):
            x,y = pol2cart(rho[__],theta[_])
            xx.append(center[0]+math.ceil(x))
            yy.append(center[1]+math.ceil(y))
        line1 = np.column_stack((xx,yy))
        Ray_mask=np.zeros(np.shape(ROI))  
        
        for __,item in enumerate(line1):
            i = item[0]
            j = item[1]
            Ray_mask[j,i] = 1
            
        Ray_masks.append(Ray_mask)

    return np.asarray(Ray_masks)
