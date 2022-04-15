from scipy.signal import convolve2d
import numpy as np

import imageio
import skimage.color
import scipy.ndimage.filters as filters

GRAYSCALE = 1
RGB = 2
FILTER_CREATOR_CONV = np.array([1, 1])
GRAY_LEVELS = 255
MIN_RES = 16


def read_image(filename, representation):
    """
    :param filename: the image file name
    :param representation: 1 for grayscale and 2 for RGB
    :return the image in filename and convert it to the wanted representation
    """
    try:
        im = (imageio.imread(filename) / GRAY_LEVELS).astype(np.float64)
        if representation == GRAYSCALE:
            return skimage.color.rgb2grey(im)
        elif representation == RGB:
            return im
        return [[]]  # in case the representation is different then 1 or 2
    except FileNotFoundError:
        return [[]]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Function that constructs a Gaussian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: an array with the pyramid and the filter vector
    """
    filter_vec = _create_filter_vec(filter_size - 2)
    pyramid = [im]
    for i in range(max_levels - 1):
        if min(pyramid[i].shape) / 2 < MIN_RES:  # stop condition
            break
        pyramid.append(_squeeze_image(pyramid[i], filter_vec))
    return pyramid, filter_vec


def _create_filter_vec(filter_levels):
    filter_vec = FILTER_CREATOR_CONV
    for _ in range(filter_levels):
        filter_vec = np.convolve(filter_vec, FILTER_CREATOR_CONV)
    return (filter_vec / np.sum(filter_vec)).reshape(1, -1)


def _blur_image(im, filter_vec):
    return filters.convolve(filters.convolve(im, filter_vec.T), filter_vec)


def _squeeze_image(cur_im, filter_vec):
    return _blur_image(cur_im, filter_vec)[::2, ::2]


def _expand_image(im, filter_vec):
    filter_vec_doubled = filter_vec * 2
    expand_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]), dtype=im.dtype)
    expand_im[::2, ::2] = im
    return _blur_image(expand_im, filter_vec_doubled)


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



