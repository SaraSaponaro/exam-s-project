import pylab as plt
import numpy as np
import math
import imageio
from scipy.signal import convolve2d
from skimage.transform import resize
from scipy import ndimage
from scipy.ndimage.filters import generic_filter
from draw_radial_line import draw_radial_lines
from define_border import define_border
from scipy.ndimage.morphology import binary_erosion
from skimage import measure
from define_border import distanza


#leggo il file
file_id_mass='result/0016p1_2_1_massonly.png'
file_id_mask='result/0016p1_2_1_mask.png'


mass_only=imageio.imread(file_id_mass)
mask_only=imageio.imread(file_id_mask)




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