import numpy as np 
import matplotlib.pyplot as plt 
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

from generate_ellipse_dataset import *


# Test ellipses dataset creation
def test_ellipses_create():
	a = ellipses(20, (128,128))
	a.create()

	plt.plot(a.value[np.random.randint(20),:,:]);plt.show()

	return a


# Test pure sinogram computation
def test_ellipses_create_sinograms():
	a = ellipses(2, (160,160))
	image = imread(data_dir + "/phantom.png", as_gray=True)
	a.value[0,:,:] = rescale(image, scale=0.4, mode='reflect', multichannel=False)
	a.value[1,:,:] = rescale(image, scale=0.4, mode='reflect', multichannel=False)

	radona0 = radon(a.value[0,:,:], circle=False)
	radona1 = radon(a.value[1,:,:], circle=False)

	a.create_sinograms()

	return np.allclose(a.sinograms, np.array([radona0, radona1]))


# Test requested measurements
def test_ellipses_request_measurement():
	a = ellipses(2, (160,160))
	image = imread(data_dir + "/phantom.png", as_gray=True)
	a.value[0,:,:] = rescale(image, scale=0.4, mode='reflect', multichannel=False)
	a.value[1,:,:] = rescale(image, scale=0.4, mode='reflect', multichannel=False)	

	S_meas = a.request_measurement(theta=np.arange(40), SNR=10.)

	plt.imshow(S_meas[0]);plt.show()
	plt.imshow(S_meas[1]);plt.show()


