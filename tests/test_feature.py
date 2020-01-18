"""
creo un cerchio per farci lo unit test
"""
import numpy as np
import pylab as plt
import unittest
from skimage.draw import circle_perimeter
from scipy import ndimage
from feature import mass_area



test=np.zeros((126,126),dtype=np.uint8)
rr, cc = circle_perimeter(63, 63, 10)
test[rr,cc]=1
circle=ndimage.binary_fill_holes(test).astype(int)


class Test_feature(unittest.TestCase):
    def mass_area(self):
        self.assertAlmostEqual(mass_area(circle),np.pi*100)
        
        





