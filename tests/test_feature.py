"""
creo un cerchio per farci lo unit test
"""
import numpy as np
import pylab as plt
import unittest
from skimage.draw import circle_perimeter
from scipy import ndimage




test=np.zeros((126,126),dtype=np.uint8)
rr, cc = circle_perimeter(63, 63, 10)
test[rr,cc]=1
fill=ndimage.binary_fill_holes(img).astype(int)

class Test_draw_radial_lines(unittest.TestCase):
    def test_feature(self):
        
        





