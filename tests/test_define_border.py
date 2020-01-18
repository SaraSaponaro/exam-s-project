import numpy as np
import random
import unittest
from segmentation_program.define_border import define_border
from segmentation_program.draw_radial_line import draw_radial_lines
from skimage.draw import circle_perimeter
from skimage import util 
from scipy import ndimage



test=np.zeros((126,126),dtype=np.uint8)
test[63-10:63+10, 63-10:63+10]=100
test[63-9:63+9, 63-9:63+9]=0
R=draw_radial_lines(test,[63,63],20,33)
a=define_border(test,33,test,5,R)
w=np.where(a==1)
ww=np.where(test==100)





class Test_feature(unittest.TestCase):
    
    def test_square(self):
        self.assertAlmostEqual(len(w[0])/100, len(ww[0])/100,1)
        
        
if __name__ == '__main__':
    unittest.main()

        
