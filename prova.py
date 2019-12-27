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


"leggo il file"
fileID='0016p1_2_1.png'
#fileID='0025p1_4_1.png'
#fileID='0036p1_1_1.png'
image=imageio.imread(fileID)
'''
plt.figure()
plt.imshow(image)
plt.show()
'''

#%% parametri
smooth_factor= 8
scale_factor= 8
size_nhood_variance=5
NL=33

#%%processo l'img
k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
im_conv=convolve2d(image, k )
im_resized = resize(im_conv, (126,126), preserve_range=True)
im_norm = im_resized/np.max(im_resized)

plt.figure('immagine normalizzata')
plt.imshow(im_norm)
plt.grid(True)
plt.show()

#%%SEGMENTATION
print('Inserisci le coordinate dela roi che contine la massa da segmentare')
y1=int(input('Inserisci il tuo y1: '))
x1=int(input('Inserisci il tuo x1: '))
y2=int(input('Inserisci il tuo y2: '))
x2=int(input('Inserisci il tuo x2: '))

''''per il momento seleziono una roi a mano dell'immagine
y1=29
y2=90
x1=30
x2=97'''

ROI=np.zeros(np.shape(im_norm))
ROI[y1:y2,x1:x2]=im_norm[y1:y2,x1:x2]
y_max,x_max=np.where(ROI==np.max(ROI))

'''if the point with maximum intensity is too far away from the ROI center, we
 chose the center of the rectangle as starting point'''
if((np.abs(x_max-(x2-x1)/2)>(4/5)*(x1+(x2-x1)/2)) or (np.abs(y_max-(y2-y1)/2)>(4/5)*(y1+(y2-y1)/2))):
    x_center=x1+int((x2-x1)/2)
    y_center=y1+int((y2-y1)/2)
else:
    x_center=x_max[0]
    y_center=y_max[0]

plt.figure('ROI')
plt.imshow(im_norm)
plt.imshow(ROI, alpha=0.5)
plt.show()

#%%radial lines
R=int(np.sqrt((x2-x1)**2+(y2-y1)**2)/2)     #intero più vicino
center=[x_center, y_center]
nhood=np.ones((size_nhood_variance,size_nhood_variance))
Ray_masks=draw_radial_lines(ROI,center,R,NL)

plt.figure('raggio casuale')
plt.imshow(Ray_masks[20])
plt.imshow(im_norm, alpha=0.5)
plt.imshow(ROI, alpha=0.5)
plt.plot(center[0],center[1], 'r.')
plt.show()

#%% border
roughborder,bordofinale_x,bordofinale_y=define_border(im_norm, NL, ROI,size_nhood_variance, Ray_masks)

plt.figure()
plt.plot(bordofinale_x,bordofinale_y,'.')
plt.imshow(roughborder)
plt.imshow(im_norm, alpha=0.5)
plt.show()

#%% imfill border
fill=ndimage.binary_fill_holes(roughborder).astype(int)

plt.figure()
plt.title('fill')
plt.imshow(fill)


#%%continua
R_raff = int(R/5)
fill_tot=[]
for _ in range(0, len(bordofinale_x)):
    center_raff = [bordofinale_x[_], bordofinale_y[_]]
    Ray_masks_raff = draw_radial_lines(ROI,center_raff,R_raff,NL)
    roughborder_raff, _ , _ = define_border(im_norm, NL, ROI,size_nhood_variance, Ray_masks_raff)
    roughborder+=roughborder_raff

fill_raff=ndimage.binary_fill_holes(roughborder).astype(int)

plt.figure()
plt.imshow(fill_raff, cmap='gray')
plt.imshow(fill, cmap='gray', alpha=0.3)
plt.show()
