import os
import numpy as np
from numpy.fft import fft2, ifft2
from skimage import io
from skimage.transform import resize
from scipy.signal import gaussian

# Function will add gaussian noise to the image
def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

# Function will return patches of default size (11,11)
def patchify(img, patch_shape):
    X, Y = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y)
    X_str, Y_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

# Function to convert RGB image to grayscale
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Creates a gaussian kernel
def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

# Wiener filter
def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), './Noisy Image/photo.png')
    gaussian_noise_image = os.path.join(os.getcwd(), './Noisy Image/gaussian_noise.png')

    myPhoto = io.imread(filename)

    # Code to create a gaussian noisy image
    #gaussian_noise = add_gaussian_noise(myPhoto, 20)
    #io.imsave('./Noisy Image/gaussian_noise.png', gaussian_noise)

    gaussian_noise_image = io.imread(gaussian_noise_image)

    kernel = gaussian_kernel(3)

    if(len(gaussian_noise_image.shape) == 3):
        noisy_image = np.copy(gaussian_noise_image)
        red_noise = noisy_image[:, :, 0]
        green_noise = noisy_image[:, :, 1]
        blue_noise = noisy_image[:, :, 2]

        red_filtered = wiener_filter(red_noise, kernel, K=10)
        green_filtered = wiener_filter(green_noise, kernel, K=10)
        blue_filtered = wiener_filter(blue_noise, kernel, K=10)
        filtered_image = np.dstack((red_filtered, green_filtered, blue_filtered))
    else:
        filtered_image = wiener_filter(gaussian_noise_image, kernel, K=10)

    io.imsave('./Restored Images/restored.png', filtered_image)

    # Creating patches
    #patches = patchify(gaussian_noise_image, (11, 11))
    #print(patches.shape)