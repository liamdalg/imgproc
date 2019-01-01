#cython: language_level=3

import numpy as np
cimport cython

DTYPE = np.double

@cython.boundscheck(False)
@cython.wraparound(False)
def _kernel_convolution_2d(double[:, :] arr, double[:, :] kernel) -> np.ndarray:
    cdef Py_ssize_t kernel_rows = kernel.shape[0], kernel_cols = kernel.shape[1]
    cdef Py_ssize_t x_offset = kernel_rows // 2, y_offset = kernel_cols // 2

    cdef double[:, :] flipped_kernel = np.fliplr(np.flipud(kernel))

    cdef Py_ssize_t i = 0, j = 0, k = 0, l = 0
    cdef Py_ssize_t height = arr.shape[0], width = arr.shape[1]

    cdef double[:, :] conv = np.empty((height, width), dtype=DTYPE)

    cdef double tmp = 0
    cdef Py_ssize_t arr_row = 0, arr_col = 0
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

def sobel_operator(double[:, :] img, double threshold) -> np.ndarray:
    cdef double[:, :] kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=DTYPE)
    cdef double[:, :] kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=DTYPE)
    cdef double[:, :] x = _kernel_convolution_2d(img, kernel_x)
    cdef double[:, :] y = _kernel_convolution_2d(img, kernel_y)

    cdef double[:, :] edges = np.empty((x.shape[0], x.shape[1]), dtype=DTYPE)
    cdef double val = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = int(((x[i,j] ** 2) + (y[i,j] ** 2)) ** 0.5)
            edges[i][j] = val if val > threshold else 0

    return np.asarray(edges)