import pylab as plt
import numpy as np
import imageio
import statistics as stat
from skimage import measure
from define_border import distanza


#leggo il file
file_id_mask='result/0016p1_2_1_mask.png'
file_id='result/0016p1_2_1_resized.png'
mask_only=imageio.imread(file_id_mask)
img=imageio.imread(file_id)

mass=img*mask_only
#%%
def mass_area(mask_only):
    a=np.where(mass_only!=0)
    area= np.shape(a)[1]
    return area 

def mass_perimetro(mask_only):
    contours = measure.find_contours(mask_only, 0)
    return np.shape(contours)[1]

"c=1 se cerchio unitest"
def circularity(area, perimetro):          
    c = 4*np.pi*area/(perimetro**2)
    return c
    
def mu_NRL(mask_only, center, perimetro):
    contours = measure.find_contours(mask_only, 0)
    arr=  contours[0]
    arr = arr.flatten('F')
    y = arr[0:99]
    x = arr[99:]
    d=distanza(center[0],center[1], x, y)
    d_m=np.max(d)
    d=d/d_m
    d_mean=np.sum(d)/perimetro
    return d, d_mean 

def sigma_NRL(d,d_mean, perimetro):
    somm=np.sum((d-d_mean)**2)
    return np.sqrt(somm/perimetro)

'''
def Radial_lenght_entropy(d):
    n, bins, p = plt.hist(d, 5)
'''

'''quante volt è maggire uguale d_mean'''
def cross_zerod(d,d_mean):
   c = np.where(d>d_mean)
   return len(c[0])

'''chiedi a luigi 
def axis(mask_only, x_b, y_b):
    for _ in range(0, len(x_b)): 
        a = distanza(x[0],y[0], x[_], y[_])
'''
'''secondo te è d=d/d_max o solo d la radial length ?'''
def VR(d), d_mean):
    v=d-d_mean
    vm=np.max(v)/2
    mean = np.mean(v>=vm)
    std = np.std(v>=vm)
    return mean, std 

'''convexity: mi serve la tua intelligenza geometrica'''

'''13-14 mean std dell'img mass_only'''
def mass_intensity(mass):
    mean=np.mean(mass)
    std=np.std(mass)
    return mean,std


'''controllare definizione di courtosi'''
def kurtosis(mean_m, std_m):
    num=np.sum(mass-mean_m)**4/(np.shape(mass)[0]*np.shape(mass)[1])
    den=std_m**4
    return num/den
    
    
    
    
    
    