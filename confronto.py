import os
import glob
from PIL import Image
import pylab as plt

#change extension
'''
confronto = glob.glob('ref/*.pgm')

for i in range(0,len(confronto),1):
    source = confronto[i]
    filename, file_extension = os.path.splitext(source)
    dest = filename+'.png'
    os.rename(source, dest) 
'''
mask = glob.glob('result/0016p1_2_1_mask.png')
confronto = glob.glob('ref/0016p1_2_1_mass_mask.png')

plt.imshow(Image.open(confronto[0]), cmap='gray')
#plt.imshow(Image.open(mask[0]), alpha=0.5, cmap='gray')
plt.imshow(Image.open('ref/0016p1_2_1_resized.png'), alpha=0.7)

