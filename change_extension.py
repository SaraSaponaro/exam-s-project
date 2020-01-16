import os
import glob
from PIL import Image
import pylab as plt

#change extension

confronto = glob.glob('/Users/luigimasturzo/Documents/esercizi_fis_med/large_sample_Im_segmented_ref/*.pgm')

for i in range(0,len(confronto),1):
    source = confronto[i]
    filename, file_extension = os.path.splitext(source)
    dest = filename+'.png'
    os.rename(source, dest)
