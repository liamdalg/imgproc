"""Image processing module.

Provides functionality for processing images,
such as edge detection, filters, etc.
"""

import numpy as np
import cv2 as cv

__version__ = '0.1'
__author__ = 'liamdalg'

def _kernel_convolution_2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Returns a convolved numpy array.

    Note that the source array is padded with zeros around the edges. This 
    ensures that the resulting array has the same shape as the source.

    Args:
        arr: 2 dimensional source array to convolve.
        kernel: 2 dimensional kernel - this is flipped during convolution.
    """

    # TODO: Optimize this please!
    rows, cols = kernel.shape
    x, y = rows // 2, cols // 2
    flipped_kernel = np.fliplr(np.flipud(kernel))

    conv = np.empty(arr.shape)
    padded_arr = np.pad(np.copy(arr), ((x, y), (x, y)), mode='constant')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            conv[i][j] = np.sum(padded_arr[i:(i + rows), j:(j + cols)] * flipped_kernel)

    return conv

# TODO: switch this to work for multiple channel (colour) images
def sobel_operator(img: np.ndarray, threshold: int) -> np.ndarray:
    """Performs the combined sobel operator for edge detection on an image.

    Args:
        img: a single channel image represented as a numpy array.
        threshold: the minimum value that a pixel is considered to be an edge.

    Returns:
        A numpy array which is the combination of the sobel operator in the x
        and y direction.
    """

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x = _kernel_convolution_2d(img, kernel_x)
    y = _kernel_convolution_2d(img, kernel_y)
    
    edges = np.empty(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = int(((x[i,j] ** 2) + (y[i,j] ** 2)) ** 0.5)
            edges[i,j] = val if val > threshold else 0
    
    return edges


def laplacian_operator(img: np.ndarray, reduce_noise=True) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edges = _kernel_convolution_2d(img, kernel)
    return edges