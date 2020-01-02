import pylab as plt
import numpy as np
import imageio
import logging
import glob
import statistics as stat
from skimage import measure
from define_border import distanza
from scipy.stats import norm, kurtosis, skew
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def linear(x1,y1,x2,y2):
    m=(y2-y1)/(x2-x1)
    q=y1 - x1*(y2-y1)/(x2-x1)
    return m,q

def mass_area(mask_only):
    a=np.where(mask_only != 0)
    area= np.shape(a)[1]
    return area

def mass_perimeter(mask_only):
    contours = measure.find_contours(mask_only, 0)
    return np.shape(contours)[1]

"c=1 se cerchio unitest"
def circularity(area, perimetro):
    c = 4*np.pi*area/(perimetro**2)
    return c

def mu_NRL(mask_only, center_x, center_y, perimetro):
    contours = measure.find_contours(mask_only, 0)
    arr=  contours[0]
    arr = arr.flatten('F')
    y = arr[0:1+int(len(arr)/2)]
    x = arr[1+int(len(arr)/2):]
    d=distanza(center_x,center_y, x, y)
    d_m=np.max(d)
    d_norm=d/d_m
    d_mean=np.sum(d_norm)/perimetro
    return d, d_mean, d_norm

def sigma_NRL(d,d_mean, perimetro):
    somm=np.sum((d-d_mean)**2)
    return np.sqrt(somm/perimetro)

'''
def Radial_lenght_entropy(d):
    n, bins, p = plt.hist(d, 5)
'''

'''quante volt è maggire uguale d_mean'''
def cross_zero(d,d_mean):
   c = np.where(d>d_mean)
   return len(c[0])

def rope(mass,mask_only, center_x, center_y):
    contours = measure.find_contours(mask_only, 0)
    arr=  contours[0]
    arr = arr.flatten('F')
    y = arr[0:1+int(len(arr)/2)]
    x = arr[1+int(len(arr)/2):]


    plt.figure('per ogni punto printo la linea che passa piu vicina al centro')
    plt.imshow(mask_only)

    l_list=[]
    for _ in range(0,len(x)):
        a_value=[]
        m_value=[]
        q_value=[]
        for __ in range(0,len(x)):

            if(x[_]!=x[__]):
                m,q=linear(x[_],y[_],x[__],y[__])
                a=center_y-m*center_x-q
                a_value.append(a)
                m_value.append(m)
                q_value.append(q)

        a_value=np.asarray(a_value)     #se è uguale a zero la retta passa per il centro
        m_value=np.asarray(m_value)
        q_value=np.asarray(q_value)

        a_value=np.abs(a_value)

        a_min_value=np.where(a_value==a_value.min())

        R=distanza(x[_],y[_],x[a_min_value[0][0]], y[a_min_value[0][0]])
        l_list.append(R)

        x_plot=np.linspace(x[_],x[a_min_value[0][0]],400)
        y_plot=m_value[a_min_value[0][0]]*x_plot + q_value[a_min_value[0][0]]
        plt.plot(x_plot,y_plot)



    return np.min(l_list), np.max(l_list)

    plt.show()


def VR(d, d_mean):
    v=d-d_mean
    vm=np.max(v)/2
    mean = np.mean(np.abs(v)>=vm)
    std = np.std(np.abs(v)>=vm)
    return mean, std

def convexity(mass,area):
    c=np.where(mass>0)
    y=c[0]
    x=c[1]
    coordinate=np.hstack((x,y))
    coordinate=coordinate.reshape(2, -1).T
    hull= ConvexHull(coordinate)
    #plt.plot(coordinate[:,0], coordinate[:,1], 'o')
    plt.plot(coordinate[hull.vertices,0], coordinate[hull.vertices,1], 'ko')
    for simplex in hull.simplices:
        plt.plot(coordinate[simplex, 0], coordinate[simplex, 1], 'r-')
    plt.imshow(mass)
    return area/hull.volume

def mass_intensity(mass):
    mean=np.mean(mass)
    std=np.std(mass)
    return mean,std



#%%

if __name__ == '__main__':
    logging.info('Reading files.')
    files=glob.glob('result/*_resized.png')
    masks=glob.glob('result/*_mask.png')
    center_x, center_y = np.loadtxt('center_list.txt', unpack=True, usecols=(1,2))
    mass_area_list=[]
    mass_perimeter_list=[]
    circularity_list=[]
    mu_NRL_list=[]
    sigma_NRL_list=[]
    cross_zero_list=[]
    rope_max_list=[]
    rope_min_list=[]
    VR_mean_list=[]
    VR_std_list=[]
    convexity_list=[]
    mass_intensity_mean_list=[]
    mass_intensity_std_list=[]
    kurtosis_list=[]
    skew_list=[]

    for _ in range(0, len(files)):
        mask_only=imageio.imread(masks[_])
        img=imageio.imread(files[_])
        mass=img*mask_only

        area=mass_area(mask_only)
        p=mass_perimeter(mask_only)

        mass_area_list.append(area)
        mass_perimeter_list.append(p)

        circularity_list.append(circularity(area,p))
        d, d_mean, __ = mu_NRL(mask_only, center_x[_], center_y[_], p)     #fai file con center

        mu_NRL_list.append(d_min)
        sigma_NRL_list.append(sigma_NRL(d, d_mean, p))
        cross_zero_list.append(cross_zero(d, d_mean))

        rmax, rmin = rope(mass, mask_only, center_x[_], center_y[_])

        rope_max_list.append(rmax)
        rope_min_list.append(rmin)

        vm, vs= VR(d, d_mean)

        VR_mean_list.append(vm)
        VR_std_list.append(vs)
        convexity_list.append(convexity(mass, area))

        im, istd = mass_intensity(mass)
        mass_intensity_mean_list.append(im)
        mass_intensity_std_list.append(istd)
        kurtosis_list.append(kurtosis(mass))
        skew_list.append(skew(mass))
