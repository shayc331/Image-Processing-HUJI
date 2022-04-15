import numpy as np
import scipy.ndimage.filters as filters
import imageio
import skimage.color
import matplotlib.pyplot as plt
import os


GRAYSCALE = 1
RGB = 2
K = 255

FILTER_CREATOR_CONV = np.array([1, 1])
MIN_RES = 16

SUBPLOT_TITLES = np.array([['First Image', 'Second Image'], ['Mask', 'Blended Image']])
FIRST_EXAMPLE_TITLE = "King In The North"
SECOND_EXAMPLE_TITLE = "Flying Giannis"


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    :param filename: the image file name
    :param representation: 1 for grayscale and 2 for RGB
    :return the image in filename and convert it to the wanted representation
    """
    try:
        im = (imageio.imread(filename) / K).astype(np.float64)
        if representation == GRAYSCALE:
            return skimage.color.rgb2grey(im)
        elif representation == RGB:
            return im
        return [[]]  # in case the representation is different then 1 or 2
    except FileNotFoundError:
        return [[]]


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Function that constructs a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: an array with the pyramid and the filter vector
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyr = list()
    for i in range(len(gaussian_pyr) - 1):
        laplacian_pyr.append(gaussian_pyr[i] - _expand_image(gaussian_pyr[i + 1], filter_vec))
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: filter vector
    :param coeff: a python list, each level i multiply with coeff[i]
    :return: Image that reconstruct from the given Laplacian pyramid
    """
    pyramid = lpyr.copy()
    for i in range(min(len(pyramid), len(coeff)) - 1, 0, -1):
        pyramid[i - 1] = _expand_image(pyramid[i], filter_vec) + (pyramid[i - 1] * coeff[i])
    return pyramid[0]


def _create_display_image_dims(pyr, levels):
    n, m = pyr[0].shape[0], 0
    for i in range(levels):
        m += pyr[i].shape[1]
    return n, m


def _normalized_pyramid(pyr):
    for i in range(len(pyr)):
        max_val, min_val = np.max(pyr[i]), np.min(pyr[i])
        pyr[i] = (pyr[i] - min_val) / (max_val - min_val) if max_val != min_val else pyr[i]  # if the val is zero dont strech


def render_pyramid(pyr, levels):
    """
    :param pyr: a Gaussian or a Laplacian pyramid
    :param levels: number of levels to present in the result
    :return: A single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    _normalized_pyramid(pyr)
    res = np.zeros(_create_display_image_dims(pyr, min(levels, len(pyr))))
    last_y = 0
    for i in range(levels):
        res[: pyr[i].shape[0], last_y: last_y + pyr[i].shape[1]] = pyr[i]
        last_y += pyr[i].shape[1]
    return res


def display_pyramid(pyr, levels):
    """
    The Function display the given pyramid
    :param pyr: a Gaussian or a Laplacian pyramid
    :param levels: number of levels to present in the result
    """
    result = render_pyramid(pyr, levels)
    plt.imshow(result, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: the first image, grayscale
    :param im2: the second image, grayscale
    :param mask: is a boolean mask containing True and False representing which parts
                of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                           defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter(an odd scalar that represents a squared filter) which
                             defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: the blending image according to the input images and the mask
    """
    L_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G_m, _ = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    L_out = [G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i] for i in range(len(L_1))]
    result = laplacian_to_image(L_out, filter_vec, np.ones(len(L_1)))
    return np.clip(result, 0., 1.)


def _blend_RGB_im(im1, im2, mask):
    im_blend = np.zeros_like(im1)
    for channel in range(3):
        im_blend[:, :, channel] = pyramid_blending(im1[:, :, channel], im2[:, :, channel], mask, 3, 3, 3)
    return im_blend


def _create_subplot(images, title):
    n, m = len(images), len(images[0])
    fig, axarr = plt.subplots(n, m, figsize=(20, 20))
    fig.suptitle(title, fontsize=60)
    for i in range(n):
        for j in range(m):
            axarr[i][j].imshow(images[i][j], cmap='gray')
            axarr[i][j].set_title(SUBPLOT_TITLES[i, j], size=44)
    plt.show()


def _convert_to_binary_im(im):
    im[im < 0.1] = 0.
    im[im >= 0.1] = 1.
    return im.astype(np.bool)


def blending_example1():
    """
    My first blending example
    :return: the first image, thw second image, the mask image and the blending image
    """
    im2 = read_image(relpath("hermon.jpg"), RGB)
    im1 = read_image(relpath("rob_stark.jpg"), RGB)
    mask = read_image(relpath("rob_mask.jpg"), GRAYSCALE).astype(np.float64)
    mask = _convert_to_binary_im(mask)
    im_blend = _blend_RGB_im(im1, im2, mask)
    _create_subplot([[im1, im2], [mask, im_blend]], FIRST_EXAMPLE_TITLE)
    return im1, im2, mask, im_blend


def blending_example2():
    """
    My second blending example
    :return: the first image, thw second image, the mask image and the blending image
    """
    im2 = read_image(relpath("helicopter.jpg"), RGB)
    im1 = read_image(relpath("giannis.jpg"), RGB)
    mask = read_image(relpath("giannis_mask.jpg"), GRAYSCALE).astype(np.float64)
    mask = _convert_to_binary_im(mask)
    im_blend = _blend_RGB_im(im1, im2, mask)
    _create_subplot([[im1, im2], [mask, im_blend]], SECOND_EXAMPLE_TITLE)
    return im1, im2, mask, im_blend

if __name__ == '__main__':
    blending_example2()
    blending_example1()