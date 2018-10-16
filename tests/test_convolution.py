import numpy as np
import unittest
import imgproc

class TestConvolutions(unittest.TestCase):
    """Tests method related to kernel convolutions.
    """

    def test_basic_3x3_conv(self):
        source = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        true_conv = np.array([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
        conv = imgproc._kernel_convolution_2d(source, kernel)

        self.assertTrue(np.array_equiv(true_conv, conv), 'Expected Value:\n{}\nActual Value:\n{}'.format(true_conv, conv))

if __name__ == '__main__':
    unittest.main()