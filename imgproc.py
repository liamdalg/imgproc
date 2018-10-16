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
    flipped_kernel = np.fliplr(np.flipud(kernel))

    conv = np.empty(arr.shape)
    padded_arr = np.pad(np.copy(arr), ((x, y), (x, y)), mode='constant')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            conv[i][j] = np.sum(padded_arr[i:(i + rows), j:(j + cols)] * flipped_kernel)

    return conv

def sobel_operator(img: np.ndarray, threshold: int) -> np.ndarray:
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x = _kernel_convolution_2d(img, sobel_x)
    y = _kernel_convolution_2d(img, sobel_y)
    cv.imwrite('outx.jpg', x)
    cv.imwrite('outy.jpg', y)
    
    combined = np.empty(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = int(((x[i,j] ** 2) + (y[i,j] ** 2)) ** 0.5)
            combined[i,j] = val if val > threshold else 0
    
    cv.imwrite('outcombined.jpg', combined)
    return img


def laplacian_operator(img: np.ndarray) -> np.ndarray:
    return img