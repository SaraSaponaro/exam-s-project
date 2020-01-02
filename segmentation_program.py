import pylab as plt
import numpy as np
import imageio
import os
import logging
from scipy.signal import convolve2d
from skimage.transform import  rescale, resize
from PIL import Image
from scipy import ndimage
from draw_radial_line import draw_radial_lines
from define_border import define_border_new


logging.info('Si legge il file.')
fileID='0016p1_2_1.png'
#fileID='0025p1_4_1.png'
fileID='0036p1_1_1.png'
#fileID='NL_4.png'

image=imageio.imread(fileID)

filename, file_extension = os.path.splitext(fileID)
path_out = 'result/'

if (os.path.exists(path_out)==False):
    logging.info ('creo cartella in cui salvare il risultato\n')
    os.makedirs('result')

file_out=path_out+filename+'_resized'+file_extension
mask_out=path_out+filename+'_mask'+file_extension

#%% parametri
logging.info('inserisco parametri per la segmentazione.')
smooth_factor= 8
scale_factor= 8
size_nhood_variance=5   #controlla di usarlo
NL=33

#%%processo l'img
logging.info('Si processa immagine.')
k = np.ones((smooth_factor,smooth_factor))/smooth_factor**2
im_conv=convolve2d(image, k )       #per ridurre il rumore (alte frequenze)
im_resized = rescale(im_conv, 1/scale_factor)
im_norm = im_resized/np.max(im_resized)

plt.figure('immagine normalizzata')
plt.imshow(im_norm)
plt.grid(True)
plt.show()

#%%SEGMENTATION
logging.info('Inizia la segmentazione dell immagine.')
print('Inserisci le coordinate dela roi che contine la massa da segmentare')
y1=int(input('Inserisci il tuo y1: '))
x1=int(input('Inserisci il tuo x1: '))
y2=int(input('Inserisci il tuo y2: '))
x2=int(input('Inserisci il tuo x2: '))

ROI=np.zeros(np.shape(im_norm))
ROI[y1:y2,x1:x2]=im_norm[y1:y2,x1:x2]
y_max,x_max=np.where(ROI==np.max(ROI))

'''se il punto con la massima intensità è troppo lontano dal centro della ROI,
si sceglie il centro della ROI come punto di partenza dei raggi'''
if((np.abs(x_max-(x2-x1)/2)>(4/5)*(x1+(x2-x1)/2)) or (np.abs(y_max-(y2-y1)/2)>(4/5)*(y1+(y2-y1)/2))):
    x_center=x1+int((x2-x1)/2)
    y_center=y1+int((y2-y1)/2)
else:
    x_center=x_max[0]
    y_center=y_max[0]
center=[x_center, y_center]
print(center)

plt.figure('ROI')
plt.imshow(im_norm)
plt.imshow(ROI, alpha=0.5)
plt.plot(center[0], center[1], 'r.')
plt.show()

#%%radial lines
R=int(np.sqrt((x2-x1)**2+(y2-y1)**2)/2)     #intero più vicino
nhood=np.ones((size_nhood_variance,size_nhood_variance))
Ray_masks=draw_radial_lines(ROI,center,R,NL)
print(np.shape(Ray_masks))

plt.figure('raggio casuale -> Ray_masks')
plt.imshow(Ray_masks[0])
plt.imshow(im_norm, alpha=0.5)
plt.imshow(ROI, alpha=0.5)
plt.plot(center[0],center[1], 'r.')
plt.show()

#%% border
roughborder,bordofinale_x,bordofinale_y=define_border_new(im_norm, NL, ROI,size_nhood_variance, Ray_masks)

plt.figure('bordo')
#plt.plot(bordofinale_x,bordofinale_y,'.')
plt.imshow(roughborder)
plt.imshow(im_norm, alpha=0.5)
plt.show()

#%% imfill border
fill=ndimage.binary_fill_holes(roughborder).astype(int)

plt.figure('maschera della massa segmentata')
plt.imshow(fill)
plt.show()
#%%iterated
from define_border import imbinarize

R_raff = int(R/5)
for _ in range(0, len(bordofinale_x)):
    center_raff = [bordofinale_x[_], bordofinale_y[_]]
    Ray_masks_raff = draw_radial_lines(ROI,center_raff,R_raff,NL)
    roughborder_raff, _ , _ = define_border_new(im_norm, NL, ROI,size_nhood_variance, Ray_masks_raff)
    roughborder+=roughborder_raff

fill_raff=ndimage.binary_fill_holes(roughborder).astype(int)

plt.figure('maschera finale')
plt.imshow(fill_raff, cmap='gray')
#plt.imshow(fill, cmap='gray', alpha=0.3)
plt.show()

#%% show result and save output
mass_only = fill_raff*im_norm
plt.figure('massa segmentata')
plt.imshow(mass_only, cmap='gray')
plt.show()

logging.info('Si salvano i risulati.')
im_resized = im_resized.astype(np.uint8)
im.save(file_out)

fill_raff = fill_raff.astype(np.int8)
im1 = Image.fromarray(fill_raff, mode='L')
im1.save(mask_out)
