import numpy as np
import unittest
import imgproc_cy

def format_expected(expected, actual):
    return 'Expected Value: {}\nActual Value: {}'.format(expected,actual)

class TestConvolutions(unittest.TestCase):
    """Tests methods related to kernel convolutions.
    """

    def test_1x1_conv(self):
        source = np.array([[1]]).astype(np.float64)
        kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]).astype(np.float64)
        true_conv = np.array([[2]]).astype(np.float64)
        conv = np.asarray(imgproc_cy._kernel_convolution_2d(source, kernel))

        self.assertTrue(np.array_equal(true_conv, conv), format_expected(true_conv, conv))

    def test_1x3_conv(self):
        source = np.array([[1], [2], [3]]).astype(np.float64)
        kernel = np.array([[1, 5, 1], [1, 5, 1], [1, 5, 1]]).astype(np.float64)
        true_conv = np.array([[15], [30], [25]]).astype(np.float64)
        conv = np.asarray(imgproc_cy._kernel_convolution_2d(source, kernel))

        self.assertTrue(np.array_equal(true_conv, conv), format_expected(true_conv, conv))

    def test_3x1_conv(self):
        source = np.array([[1, 2, 3]]).astype(np.float64)
        kernel = np.array([[1, 1, 1], [10, 10, 10], [1, 10, 1]]).astype(np.float64)
        true_conv = np.array([[30, 60, 50]]).astype(np.float64)
        conv = np.asarray(imgproc_cy._kernel_convolution_2d(source, kernel))

        self.assertTrue(np.array_equal(true_conv, conv), format_expected(true_conv, conv))

    def test_3x3_conv(self):
        source = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float64)
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float64)
        true_conv = np.array([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]]).astype(np.float64)
        conv = np.asarray(imgproc_cy._kernel_convolution_2d(source, kernel))

        self.assertTrue(np.array_equal(true_conv, conv), format_expected(true_conv, conv))

    def test_4x3_conv(self):
        source = np.array([[1, 5, 2, 3], [8, 7, 3, 6], [3, 3, 9, 1]]).astype(np.float64)
        kernel = np.array([[1, 2, 3], [0, 0, 0], [6, 5, 4]]).astype(np.float64)
        true_conv = np.array([[23, 41, 33, 21], [44, 65, 76, 52], [82, 85, 79, 42]]).astype(np.float64)
        conv = np.asarray(imgproc_cy._kernel_convolution_2d(source, kernel))

        self.assertTrue(np.array_equal(true_conv, conv), format_expected(true_conv, conv))


if __name__ == '__main__':
    unittest.main()