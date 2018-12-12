import numpy as np 
import matplotlib.pyplot as plt 
import prox_tv
import pickle
import scipy.linalg as la
from skimage.transform import radon, iradon

from generate_ellipse_dataset import *


def ist_reconstruction(Y, x_init, lamda=0.01, step=0.01, n_iter=100, verbose=1, lamda_rate=1.):

	""" Iterative Shrinkage and Thresholding Based reconstruction from sparse views CT.

	Inputs:
	`X` : An instance of class `ellipses`.
			`X.value` contains initial value of estimate. Use all the methods of `X` for getting he radon transform, FBP and backprojection. 

	`lamda` : Regularization parameter
	`n_iter`: Number of iterations
	`X_init`: Initialization
	"""

	# assert X.measurement != None, 'Please specify the low view measurement'

	x = x_init.copy()
	theta = np.linspace(0.,179., Y.shape[1])

	for i in range(n_iter):
		z = Y - radon(x, theta, circle=False)
		grad = -iradon(z, theta, circle=False, filter=None)

		x = prox_tv.tv1_2d(x - step * grad, lamda)
		lamda = lamda*lamda_rate

		if verbose==1 and i%10==0:
			print('Loss = ', la.norm(z))



	x_est = x

	return x_est






