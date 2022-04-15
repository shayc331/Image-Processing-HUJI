import numpy as np
import imageio
import skimage.color
import matplotlib.pyplot as plt

GRAYSCALE = 1
RGB = 2
RGB_CHANNELS = 3
GRAYSCALE_CHANNELS = 2
K = 255

YIQ_BASE_MAT = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]], dtype=np.float64)
RGB_BASE_MAT = np.linalg.inv(YIQ_BASE_MAT)


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


def imdisplay(filename, representation):
    """
    display the image in filename in the wanted representation
    :param filename: the image file name
    :param representation: 1 for grayscale and 2 for RGB
    """
    try:
        im = read_image(filename, representation)
        plt.imshow(im, cmap='gray')
        plt.show()
    except FileNotFoundError:
        return


def rgb2yiq(im_rgb):
    """
    :param im_rgb: an image in RGB representation
    :return: the input image in YIQ representation
    """
    try:
        return im_rgb @ YIQ_BASE_MAT
    except Exception:  # in case the image doesnt have 3 channels
        return


def yiq2rgb(im_yiq):
    """
    :param im_yiq: an image in YIQ representation
    :return: the input image in RGB representation
    """
    try:
        return im_yiq @ RGB_BASE_MAT
    except Exception:  # in case the image doesnt have 3 channels
        return


def histogram_equalize(im_orig):
    """
    perform the algorithm of histogram equalize on the input image
    :param im_orig: the input image
    :return: a list that contains the equalize image, the original histogram and the equalized histogram.
    """
    if len(im_orig.shape) == RGB_CHANNELS:
        return _histogram_equalize_rgb(im_orig)
    elif len(im_orig.shape) == GRAYSCALE_CHANNELS:
        return _histogram_equalize_grayscale(im_orig)


def _histogram_equalize_rgb(im_orig):
    im_orig = rgb2yiq(im_orig)
    im_eq, hist_orig, hist_eq = _histogram_equalize_grayscale(im_orig[:, :, 0])
    im_orig[:, :, 0] = im_eq
    im_eq_full = yiq2rgb(im_orig)
    return [im_eq_full, hist_orig, hist_eq]


def _histogram_equalize_grayscale(im):
    im = (im * K).astype(np.float64)
    hist_orig, edges = np.histogram(im, bins=K + 1, range=(0.0, 255.0))
    cum_hist_orig = np.cumsum(hist_orig)
    c_m = np.nonzero(hist_orig)[0][0]
    norm_cum_hist = np.round(np.multiply(np.divide(cum_hist_orig - c_m, cum_hist_orig[K] - c_m), K))
    T = np.vectorize(lambda x: norm_cum_hist[min(int(x), K)])
    im_eq = T(im)
    hist_eq, edges = np.histogram(im_eq, bins=K + 1, range=(0.0, 255.0))
    im_eq = (im_eq / K).astype(np.float64)
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    perform the algorithm of histogram quantize on the input image
    :param im_orig: the input image
    :return: a list that contains the quantized image and an array of errors for each iteration.
    """
    if len(im_orig.shape) == RGB_CHANNELS:
        return _quantize_rgb_dims(im_orig, n_quant, n_iter)
    elif len(im_orig.shape) == GRAYSCALE_CHANNELS:
        return _quantize_grayscale(im_orig, n_quant, n_iter)


def _quantize_rgb_dims(im_orig, n_quant, n_iter):
    im_orig = rgb2yiq(im_orig)
    im_qu, errors = _quantize_grayscale(im_orig[:, :, 0], n_quant, n_iter)
    im_orig[:, :, 0] = im_qu
    im_qu_full = yiq2rgb(im_orig)
    return [im_qu_full, errors]


def _init_z_and_q(cum_hist_orig, n_quant, segment_pixels):
    z = np.zeros(n_quant + 1).astype(np.int)
    z[-1] = K
    z[0] = -1
    for j in range(1, n_quant):  # get approximately the same amount of pixels in each segment
        z[j] = np.argmax(cum_hist_orig > segment_pixels * j)
    q = np.array([(z[i] + 1 + z[i + 1]) // 2 for i in range(n_quant)])
    return z, q


def _create_quant_im(im_orig, z, q, n_quant):
    for j in range(n_quant):
        im_orig = np.where(np.logical_and(z[j + 1] >= im_orig, z[j] + 1 <= im_orig), q[j], im_orig)
    return (im_orig / K).astype(np.float64)


def _quantize_grayscale(im_orig, n_quant, n_iter):

    def _calculate_q():
        q[0] = int(np.arange(int(z[1]) + 1) @ hist_orig[np.arange(int(z[1]) + 1)].T / cum_hist_orig[int(z[1])])
        for i in range(1, n_quant):
            sum_of_pixel_in_range = cum_hist_orig[int(z[i + 1])] - cum_hist_orig[int(z[i])]
            grey_level_range = np.arange(int(z[i]) + 1, int(z[i + 1]) + 1)
            q[i] = int((grey_level_range @ hist_orig[grey_level_range].T) / sum_of_pixel_in_range)

    def _calculate_z():
        temp_z = z.copy()
        for i in range(1, n_quant):
            temp_z[i] = int((q[i - 1] + q[i]) // 2)
        return temp_z

    def _calculate_error():
        temp_err = 0.0
        for i in range(n_quant):
            temp_err += np.square((np.arange(int(z[i]) + 1, int(z[i + 1]) + 1) * -1) + int(q[i])) @ hist_orig[int(z[i]) + 1: int(z[i + 1]) + 1]
        return temp_err

    errors = np.zeros(n_iter)
    im_orig = np.round(im_orig * K).astype(np.float64)
    hist_orig, edges = np.histogram(im_orig, bins=K + 1, range=(0.0, 255.0))
    cum_hist_orig = np.cumsum(hist_orig)
    z, q = _init_z_and_q(cum_hist_orig, n_quant, int(im_orig.size // n_quant))
    for j in range(n_iter):
        new_z = _calculate_z()
        if np.array_equal(z, new_z):
            if j == 0:
                errors = [_calculate_error()]
                break
            errors = errors[:j]
            break
        z = new_z
        _calculate_q()
        errors[j] = _calculate_error()
    quant_im = _create_quant_im(im_orig, z, q, n_quant)
    return [quant_im, errors]
