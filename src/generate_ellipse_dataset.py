import numpy as np 
import random
from numpy import cos, sin
from skimage.transform import radon
import scipy.linalg as la

""" Contains the code to generate dataset of (random ellipsod, sinogram) pairs for training the CNN
"""

class ellipses:
	""" Defined in ``src/generate_ellipse_dataset.py``

	Creates a dataset of random ellipses

	Each ellipse has an equation : 
	( (x-xi)*cos(theta) + (y-yi)*sin(theta) )**2 / a**2 + 
	( (x-xi)*sin(theta) - (y-yi)*cos(theta) )**2 / b**2 = 1 


	Attributes:
	``shape``            : Shape of dataset
	``rangeNumEllipses`` : range of number of ellipses in each image
	``value``            : numpy ndarry containing all the images
	``sinograms``        : Pure high resolution sinograms of all the images

	Methods:
	``ellipses.create()``
	``ellipses.create_sinograms()``
	``ellipses.request_measurement()``
	``ellipses.save_images()``

	"""

	# Attributes
	def __init__(self, N, image_shape):
		self.shape = (N, image_shape[0], image_shape[1])
		self.numEllipses = None
		self.value = np.zeros(self.shape)
		self.sinograms = None

	# Methods
	def create(self, N_ell=[10,20], rng_a=[0.1, 0.6], rng_b=[0.1, 0.6]):

		""" Creates the ellipses dataset. 
		
		Inputs:
		``N_ell`` : Range of number of ellipses in each image
		``rng_a`` : range of major axis values between [0,1] relative to image size
		``rng_b`` : range of minor axis values between [0,1] relative to image size

		Outputs:
		numpy ndarry containing ellipse images
		"""

		N = self.shape[0]
		images = self.value
		image_shape = self.shape[1::]

		for i in range(N):
			# generate major and minor axes
			num_ellipses = np.random.randint(N_ell[0], N_ell[1])
			a = np.random.rand(num_ellipses)*(rng_a[1]-rng_a[0]) + rng_a[0]
			a = a*min(image_shape)
			b = np.random.rand(num_ellipses)*(rng_b[1]-rng_b[0]) + rng_b[0]
			b = b*min(image_shape)

			# generate ellipse centers
			assert min(image_shape)>num_ellipses, "Image shape must be greater than the number of ellipses in each image"
			xc = random.sample(range(image_shape[0]), num_ellipses)
			yc = random.sample(range(image_shape[1]), num_ellipses)

			# generate ellipse angles
			thetas = np.random.rand(num_ellipses)*np.pi

			# create ellipses
			x,y = np.meshgrid( range(image_shape[0]), range(image_shape[1]) )
			for j in range(num_ellipses):
				eqn = ( (x-xc[j])*cos(thetas[j]) + (y-yc[j])*sin(thetas[j]) )**2 / a[j]**2 + ( (x-xc[j])*sin(thetas[j]) - (y-yc[j])*cos(thetas[j]) )**2 / b[j]**2 <= 1 
				images[i,:,:] += np.random.rand() * eqn 

		# self.value = images
		return images
			

	def create_sinograms(self, theta=np.array([None])):
		""" Create pure high view sinograms of each of the images
		Inputs : 
		``self.value`` 
		``theta`` : range of angles for taking the radon transfomr

		Outputs :
		Sinograms of all the images
		"""

		N = self.shape[0]
		image_shape = self.shape[1::]
		images = self.value
		if (theta==None).all():
			theta = np.linspace(0,179, max(*image_shape, 180) )

		sinograms = [radon(images[i,:,:], theta=theta, circle=False) for i in range(N)]
		sinograms = np.array(sinograms)

		self.sinograms = sinograms
		return sinograms


	def request_measurement(self, theta=np.array([None]), SNR=np.inf):
		""" Request measurement for a particular low view CT with noise
		Inputs : 
		``self.value`` 
		``theta`` : range of angles for taking the radon transfomr. If unspecified, the program picks linearly spaced thetas between 0 and max(180, max(image_size))
		``SNR`` : Signal to noise ratio in dB. default is ``inf`` 

		Outputs :
		Sinograms of all the images
		"""


		sinograms = self.create_sinograms(theta)
		N = sinograms.shape[0]
		image_shape = sinograms.shape[1::]
		normalization = np.sqrt(np.prod(image_shape))

		noise_levels = [la.norm(sinograms[i,:,:])*10**(-0.05*SNR)/normalization for i in range(N)]
		
		S_measured = [sinograms[i,:,:] + noise_levels[i]*np.random.random_sample(image_shape) for i in range(N)]

		return np.array(S_measured)








































