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
from define_border import define_border
import unittest

'immagini che contengono il risultato di quello che io mi aspetto'
test1_result_name='NL_4.png'
test1_result=imageio.imread(test1_result_name)

'immagini su cui fare la prova'
test1=np.zeros((4,126,126))


class Test_draw_radial_lines(unittest.TestCase):
    def test_raggi(self):
        'testo se gli do unimmagine input semplice che mi disegni correttamente le righe'
        self.assertAlmostEqual(draw_radial_lines(test1,[63,63],63,4),test1_result)