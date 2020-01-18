"""
creo un cerchio per farci lo unit test
"""
import numpy as np
import pylab as plt
import unittest
from skimage.draw import circle_perimeter
from scipy import ndimage
from segmentation_program.feature import mass_area, mass_perimeter,circularity, axis, convexity, mass_intensity



test=np.zeros((126,126),dtype=np.uint8)
rr, cc = circle_perimeter(63, 63, 10)
test[rr,cc]=1
circle=ndimage.binary_fill_holes(test).astype(int)
w=np.where(circle==1)


class Test_feature(unittest.TestCase):
    
    def test_area(self):
        self.assertAlmostEqual(mass_area(circle),len(w[0]))
        
    def test_perimeter(self):
        self.assertAlmostEqual(mass_perimeter(circle),61)
        
    def test_circularity(self):
        self.assertAlmostEqual(circularity(mass_area(circle), mass_perimeter(circle)),1,0)
    
    
    def test_axis(self):
        a,b=axis(circle)
        self.assertAlmostEqual(np.array([a,b]).all(),np.array([22,22]).all(), 0)
        
    def test_convexity(self):
        self.assertAlmostEqual(convexity(circle, mass_area(circle)), 1, 0)     
    
    def test_intensity(self):
        m,s=mass_intensity(circle)
        mean=mass_area(circle)/(126*126)
        std=np.sqrt(np.sum((circle-mean)**2)/(126*126))
        self.assertAlmostEqual(np.array([m,s]).all(),np.array([mean,std]).all())
        
    
if __name__ == '__main__':
    unittest.main()
