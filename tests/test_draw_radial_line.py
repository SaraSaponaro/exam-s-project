import numpy as np
import unittest
from segmentation_program.draw_radial_line import draw_radial_lines



test=np.zeros((126,126))
test[63][63]=1

test_n=test.copy()
test_n[1:63,63]=1

test_s=test.copy()
test_s[64:,64]=1

test_o=test.copy()
test_o[64,1:63]=1

test_e=test.copy()
test_e[63,63:]=1

lista_test=[test_e, test_s, test_o, test_n, test_e]
lista_test=np.asarray(lista_test)


class Test_draw_radial_lines(unittest.TestCase):
    def test_raggi(self):
        'testo se gli do unimmagine input semplice che mi disegni correttamente le righe'
        self.assertAlmostEqual(draw_radial_lines(test,[63,63],63,5).all(),lista_test.all())


if __name__ == '__main__':
    unittest.main()
