import convolution as conv
import colorsys
import math
import cv2
import numpy as np


def magnitude(x, y):
    return ((x ** 2) + (y ** 2)) ** 0.5


def gradient_to_rgb(x: int, y: int) -> (int, int, int):
    """
        Converts x and y gradient value into an RGB value, depending on the ratio between y and x.
    """
    if x != 0:
        # normalize from the range [-0.5pi, 0.5pi] to [0, 1]
        hue = 0.5 + (math.atan(y / x) / math.pi)
    else:
        hue = 0

    return colorsys.hsv_to_rgb(hue, 1, 0.75)


def sobel_gradient(x: np.ndarray, y: np.ndarray, threshold: int) -> np.ndarray:
    """
        Calculates the gradient for a particular pixel and colours it depending on the angle it
        makes with on a colour wheel. Positive angles are blue-ish while negative angles are
        redd-ish.
    """
    edges = np.empty((x.shape[0], x.shape[1], 3))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if magnitude(x[i, j], y[i, j]) > threshold:
                colour = gradient_to_rgb(x[i, j], y[i, j])
                edges[i, j, 0] = colour[0] * 255
                edges[i, j, 1] = colour[1] * 255
                edges[i, j, 2] = colour[2] * 255

    return edges


def sobel_threshold(x: np.ndarray, y: np.ndarray, threshold: int) -> np.ndarray:
    """
        Regular sobel operator with a threshold enforced to remove 'noise'.
    """
    edges = np.empty((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = magnitude(x[i, j], y[i, j])
            edges[i][j] = val if val > threshold else 0
    return edges


def sobel_operator(img: np.ndarray, threshold: int, gradient: bool) -> np.ndarray:
    """
        Calculates the edges of a given single-channel image.

        img: the image as a numpy array with shape (y, x)
        threshold: the minimum magnitude of a pixel to be an edge (recommend 70)
        gradient: whether the colour the edges according to their angle
    """
    kernel_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float64)
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=np.float64)
    # TODO: support fused-types instead of fixing to double/float64
    typed = img.astype(np.float64)
    x = conv.kernel_convolution_2d(typed, kernel_x)
    y = conv.kernel_convolution_2d(typed, kernel_y)

    if gradient:
        return sobel_gradient(x, y, threshold)
    else:
        return sobel_threshold(x, y, threshold)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """
        Blurs the image using a Gaussian kernel.
    """
    # TODO: make this a dynamically sized kernel
    kernel = (1 / 256) * np.array([[1,  4,  6,  4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1,  4,  6,  4, 1]], dtype=np.float64)
    typed = img.astype(np.float64)
    blurred = conv.kernel_convolution_2d(typed, kernel)
    return blurred


def _generate_examples():
    circ = cv2.imread('examples/circle.jpg', 0)
    valve = cv2.imread('examples/valve.png', 0)
    circ_out = sobel_operator(circ, 70, True)
    valve_out = sobel_operator(valve, 70, True)
    cv2.imwrite('examples/circle-grad.jpg', circ_out)
    cv2.imwrite('examples/valve-grad.png', valve_out)
