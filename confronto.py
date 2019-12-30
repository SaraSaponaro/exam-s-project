import os
import glob
from PIL import Image
import pylab as plt

#change extension
'''
confronto = glob.glob('ref/*_mask.pgm')
for i in range(0,len(confronto),1):
    source = confronto[i]
    filename, file_extension = os.path.splitext(source)
    dest = 'ref/'+filename+'.png'
    os.rename(source, dest) 
'''

mask = glob.glob('result/*_mask.png')

filename, file_extension = os.path.splitext(source)

confronto = glob.glob('ref/*_mask.png')

plt.imshow(Image.open(confronto[0]))
#plt.imshow(mask[0])