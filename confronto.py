import os
import glob
from PIL import Image
import pylab as plt

#change extension

confronto = glob.glob('/Users/sarasaponaro/Desktop/exam_cmpda/large_sample_Im_segmented_ref/*.pgm')

for i in range(0,len(confronto),1):
    source = confronto[i]
    filename, file_extension = os.path.splitext(source)
    dest = filename+'.png'
    os.rename(source, dest)

'''


mask = glob.glob('result/0069p1_4_2_mask_r10.png')
confronto = glob.glob('result/0069p1_4_2_mask.png')
loro=glob.glob('ref/0036p1_1_1_mass_mask.png')

plt.imshow(Image.open(confronto[0]), cmap='gray', label='R/5')
plt.imshow(Image.open(mask[0]), alpha=0.5, cmap='gray', label='R/10')
plt.imshow(Image.open(loro[0]), alpha=0.3, cmap='hot', label='retico')
#plt.imshow(Image.open('ref/0036p1_1_1_resized.png'), alpha=0.7)
plt.legend()
plt.show()
'''
