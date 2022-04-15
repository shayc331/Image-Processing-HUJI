import numpy as np
import imageio
import skimage.color
import scipy.io.wavfile as wav_file
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

CHANGE_RATE_OUTPUT_NAME = "change_rate.wav"
CHANGE_SAMPLE_OUTPUT_NAME = "change_samples.wav"
GRAYSCALE = 1
RGB = 2
K = 255
INVERSE = 1
NOT_INVERSE = -1

dx = np.array([[0.5, 0, -0.5]])
dy = dx.T
C = np.pi * 2j

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


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def _create_fourier_basis(signal, is_inverse):
    N = len(signal)
    samples_range = np.arange(N)
    xx, yy = np.meshgrid(samples_range, samples_range)
    indices_map = np.multiply(xx, yy)
    return np.exp([(C * is_inverse * indices_map) / N])


def DFT(signal):
    """
    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: the DFT of the signal. an array of dtype complex128 with the same shape
    """
    fourier_basis = _create_fourier_basis(signal, NOT_INVERSE)
    return (signal.T @ fourier_basis).reshape(signal.shape)



def IDFT(fourier_signal):
    """
    :param fourier_signal: an array of dtype complex128
    :return: the IDFT of the fourier_signal.
    """
    inverse_fourier_basis = _create_fourier_basis(fourier_signal, INVERSE)
    return (fourier_signal.T @ inverse_fourier_basis).reshape(fourier_signal.shape) / len(fourier_signal)



def DFT2(image):
    """
    :param image: grayscale image of dtype float64
    :return: the DFT2 of the image, 2D array of dtype complex128
    """
    image = np.apply_along_axis(DFT, 0, image)
    fourier_image = np.apply_along_axis(DFT, 1, image)
    return fourier_image



def IDFT2(fourier_image):
    """
    :param fourier_image: 2D array of dtype complex128
    :return: the IDFT2 of the fourier_image.
    """
    fourier_image = np.apply_along_axis(IDFT, 0, fourier_image)
    real_image = np.apply_along_axis(IDFT, 1, fourier_image)
    return real_image


def change_rate(filename, ratio):
    """
    function that changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    """
    rate, data = wav_file.read(filename)
    wav_file.write(CHANGE_RATE_OUTPUT_NAME, int(ratio * rate), data)


def change_samples(filename, ratio):
    """
    fast forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: a 1D ndarray of dtype float64 representing the new sample points.
    """
    rate, data = wav_file.read(filename)
    resized_data = np.real(resize(data, ratio))
    wav_file.write(CHANGE_SAMPLE_OUTPUT_NAME, rate, resized_data)
    return resized_data


def resize(data, ratio):
    """
    resize the data according to the ratio
    :param data: a 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: a positive float64 representing the duration change.
    :return: a 1D ndarray of the dtype of data representing the new sample points.
    """
    shifted_fourier = np.fft.fftshift(DFT(data))
    if ratio < 1:
        resized_data = _expand_data(shifted_fourier, ratio)
    elif ratio > 1:
        resized_data = _shrink_data(shifted_fourier, ratio)
    else:
        resized_data = shifted_fourier
    return IDFT(np.fft.ifftshift(resized_data))


def _shrink_data(data, ratio):
    length = len(data)
    to_remove, reminder = divmod(length - (length // ratio) - 1, 2)
    shrunken_data = data[int(to_remove + reminder): int((-1 * to_remove) - 1)]
    return shrunken_data


def _expand_data(data, ratio):
    length = len(data)
    to_add, reminder = divmod(np.floor(length / ratio - length).astype(np.int), 2)
    expanded_data = np.concatenate((np.zeros(int(to_add + reminder)), data, np.zeros(int(to_add))))
    return expanded_data


def resize_spectrogram(data, ratio):
    """
    a function that speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the new sample points according to ratio with the same datatype as data
    """
    spec = stft(data)
    resized_spec = np.apply_along_axis(resize, 1, spec, ratio)
    resized_data = istft(resized_spec)
    return resized_data


def resize_vocoder(data, ratio):
    """
    a function that speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the given data rescaled according to ratio with the same datatype as data
    """
    return istft(phase_vocoder(stft(data), ratio))


def conv_der(im):
    """
    function that computes the magnitude of image derivatives using convolution
    :param im: a grayscale image of type float64
    :return: a grayscale image of type float64 of the magnitude
    """
    im_dx = signal.convolve2d(im, dx, mode='same')
    im_dy = signal.convolve2d(im, dy, mode='same')
    return _calculate_magnitude(im_dx, im_dy)


def _calculate_magnitude(im_dx, im_dy):
    return np.sqrt(np.abs(im_dx) ** 2 + np.abs(im_dy) ** 2)


def _fourier_der_axis(im, axis):
    length = im.shape[axis]
    freq_range = np.arange(-1 * length * 0.5, length * 0.5)
    f = lambda x: np.multiply(x, freq_range)
    im_der = np.apply_along_axis(f, axis, im) * C / length
    return IDFT2(np.fft.ifftshift(im_der))


def fourier_der(im):
    """
    function that computes the magnitude of image derivatives using fourier
    :param im: a grayscale image of type float64
    :return: a grayscale image of type float64 of the magnitude
    """
    centered_fourier = np.fft.fftshift(DFT2(im))
    im_der_x = _fourier_der_axis(centered_fourier, 0)
    im_der_y = _fourier_der_axis(centered_fourier, 1)
    return _calculate_magnitude(im_der_x, im_der_y)

