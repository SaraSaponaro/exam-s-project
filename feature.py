import pylab as plt
import numpy as np
import imageio
from skimage import measure
from define_border import distanza


#leggo il file
file_id_mass='prova.png' #'result/0016p1_2_1_massonly.png'
file_id_mask='result/0016p1_2_1_mask.png'


mass_only=imageio.imread(file_id_mass)
mask_only=imageio.imread(file_id_mask)

#%%
def mass_area(mass_only):
    a=np.where(mass_only!=0)
    area= np.shape(a)[1]
    return area 

def mass_perimetro(mass_only):
    contours = measure.find_contours(mass_only, 1)
    return np.shape(contours)[1]

"c=1 se cerchio unitest"
def circularity(area, perimetro):          
    c = 4*np.pi*area/(perimetro**2)
    return c
    
def mu_NRL(mass_only, center):
    contours = measure.find_contours(mass_only, 1)
    for n, contour in enumerate(contours):
        x=int(contour[:, 1])
        y=int(contour[:, 0])
        d= distanza(center[0], center[1],x, y)
'''prova 
        
contours = measure.find_contours(mass_only, 1)
np.shape(contours)[1]
'''  