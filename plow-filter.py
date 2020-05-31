import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.feature_extraction import image

# Function will add gaussian noise to the image
def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

# Function will return patches of default size (11,11)
def patchify(img, patch_shape):
    X, Y, Z = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y)
    X_str, Y_str, Z_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), './Noisy Image/photo.png')
    gaussian_noise_image = os.path.join(os.getcwd(), './Noisy Image/gaussian_noise.png')

    myPhoto = io.imread(filename)

    # Code to create a gaussian noisy image
    #gaussian_noise = add_gaussian_noise(myPhoto, 20)
    #io.imsave('./Noisy Image/gaussian_noise.png', gaussian_noise)

    gaussian_noise_image = io.imread(gaussian_noise_image)

    # Creating patches
    patches = patchify(myPhoto, (11, 11))
    print(patches.shape)