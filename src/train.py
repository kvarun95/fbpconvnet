import numpy as np 
import tensorflow as tf 
from generate_ellipse_dataset import *
import tensorflow.keras.backend as K 


def load_dataset(train_size, test_size, which='ellipses', forward='radon'):

	""" Defined in /src/train.py

	Loads the dataset to be used for training and testing.
	
	Inputs : 
	`train_size` : Size of training set
	`test_size`  : size of test set
	`which` : String. specifies the type of dataset to be generated:
		`'ellipses'`   : Ellipses dataset
		`'biomedical'` : Biomedical dataset

	`forward` : String. The forward model to be used 
		`'radon'`		: Radon transform. inbuilt in the object ``ellipses``.
	"""

	if which=='ellipses':
		E = ellipses(train_size+test_size, (128,128))
		E.create(normalize='True');
		E.request_measurement(theta=np.linspace(0., 179., 20), SNR=np.inf);
		E.fbp_reconstruction();

		X_train = E.recon[0:train_size].reshape(*(E.recon[0:train_size].shape), 1)
		X_test = E.recon[train_size::].reshape(*(E.recon[train_size::].shape), 1)
		
		E_train = E.value[0:train_size].reshape(*(E.value[0:train_size].shape), 1)
		E_test = E.value[train_size::].reshape(*(E.value[train_size::].shape), 1)

	return X_train, X_test, E_train, E_test


def euclidean_loss(y_true, y_pred):
	""" 
	Defines euclidean loss between the true and estimated image
	Inputs : 
	`y_true` : True image
	`y_pred` : Estimate of the image.
	"""

	# N = y_true.shape[0]
	# print(N)
	# print(y_pred.shape)

	# assert N==y_pred.shape[0] , 'Euclidean loss: The shapes of true and estimated images must be the same.'
	
	# y_true_flat = K.reshape(y_true, (N,-1))
	# y_pred_flat = K.reshape(y_pred, (N,-1))

	return K.sqrt(K.sum(K.sum(K.sum(K.square(y_pred - y_true), axis=-1) ,axis=-1), axis=-1))




