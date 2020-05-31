import os
import numpy as np
from skimage import io
from sklearn.feature_extraction import image

# Function will add gaussian noise to the image
def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), './Noisy Image/photo.jpg')
    myPhoto = io.imread(filename)
    myPhoto = add_gaussian_noise(myPhoto, 100)
    io.imsave('./Noisy Image/gaussian_noise.png', myPhoto)