"""Image processing module.

Provides functionality for processing images,
such as edge detection, filters, etc.
"""

import numpy as np
import cv2 as cv

__version__ = '0.1'
__author__ = 'liamdalg'

def _kernel_convolution_2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows, cols = kernel.shape
    x, y = rows // 2, cols // 2

    conv = np.empty(arr.shape)
    padded_arr = np.pad(np.copy(arr), ((x, y), (x, y)), mode='constant')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            conv[i][j] = np.sum(padded_arr[i:(i + rows), j:(j + cols)] * kernel)

    return conv
    return img


def laplacian_operator(img):
    return img
