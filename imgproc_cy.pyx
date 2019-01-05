#cython: language_level=3

import numpy as np
import math 
import cv2
import colorsys
cimport cython

DTYPE = np.double

@cython.boundscheck(False)
@cython.wraparound(False)
def _kernel_convolution_2d(double[:, :] arr, double[:, :] kernel) -> np.ndarray:
    cdef Py_ssize_t kernel_rows = kernel.shape[0], kernel_cols = kernel.shape[1]
    cdef Py_ssize_t height = arr.shape[0], width = arr.shape[1]
    cdef Py_ssize_t x_offset = kernel_rows // 2, y_offset = kernel_cols // 2
    cdef Py_ssize_t i = 0, j = 0, k = 0, l = 0
    cdef Py_ssize_t arr_row = 0, arr_col = 0

    cdef double[:, :] flipped_kernel = np.fliplr(np.flipud(kernel))
    cdef double[:, :] conv = np.empty((height, width), dtype=DTYPE)
    cdef double tmp = 0
    for i in range(height):
        for j in range(width):
            for k in range(kernel_rows):
                for l in range(kernel_cols):
                    arr_col = j + l - y_offset
                    arr_row = i + k - x_offset
                    if 0 <= arr_row < height and 0 <= arr_col < width:
                        tmp += arr[arr_row, arr_col] * flipped_kernel[k, l]
            conv[i, j] = tmp
            tmp = 0

    return np.asarray(conv)

def __gradient_to_rgb(x, y):
    cdef double val = 0
    if x != 0:
        # normalise to between 0 and 1
        val = 0.5 + (math.atan(y / x) / math.pi)
    else: val = 0
    return colorsys.hsv_to_rgb(val, 1, 0.5)

def _sobel_gradient(double[:, :] x, double[:, :] y, double threshold) -> np.ndarray:
    cdef double[:, :, :] edges = np.zeros((x.shape[0], x.shape[1], 3), dtype=DTYPE)
    cdef double val = 0
    cdef Py_ssize_t i = 0, j = 0, height = x.shape[0], width = x.shape[1]
    for i in range(height):
        for j in range(width):
            if ((y[i, j] ** 2) + (x[i, j] ** 2)) ** 0.5 > threshold:
                colours = __gradient_to_rgb(x[i, j], y[i, j])
                edges[i, j, 0] = colours[0] * 255
                edges[i, j, 1] = colours[1] * 255
                edges[i, j, 2] = colours[2] * 255
    
    return np.asarray(edges)

def _sobel_threshold(double[:, :] x, double[:, :] y, threshold) -> np.ndarray:
    cdef double[:, :] edges = np.empty((x.shape[0], x.shape[1]), dtype=DTYPE)
    cdef double val = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = int(((x[i,j] ** 2) + (y[i,j] ** 2)) ** 0.5)
            edges[i][j] = val if val > threshold else 0
    return np.asarray(edges)

def sobel_operator(double[:, :] img, double threshold, bint gradient) -> np.ndarray:
    cdef double[:, :] kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=DTYPE)
    cdef double[:, :] kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=DTYPE)
    cdef double[:, :] x = _kernel_convolution_2d(img, kernel_x)
    cdef double[:, :] y = _kernel_convolution_2d(img, kernel_y)

    if gradient:
        return _sobel_gradient(x, y, threshold)
    else:
        return _sobel_threshold(x, y, threshold)

def make_example():
    img = cv2.imread('examples/circle.jpg', 0).astype(np.float64)
    out = sobel_operator(img, 70, True)
    cv2.imwrite('examples/circle-grad.jpg', out)