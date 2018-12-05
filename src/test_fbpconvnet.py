''' Tests for class fbpconvnet '''

import numpy as np 
import tensorflow as tf 

from fbpconvnet import *


def toy_data():
	''' A small tensor containing some toy images
	'''

	a = np.zeros((20, 64, 64, 3), dtype=np.float32)
	image = np.outer(np.linspace(0.,1.,64), np.linspace(0.,1.,64))

	for i in range(20):
		a[i,:,:,0] = image
		a[i,:,:,1] = image
		a[i,:,:,2] = image

	return a


def test_fbpconvnet():

	''' A small test for running fbpconvnet 
	'''

	tf.reset_default_graph()
	input_size = 64

	fbpcnn = fbpconvnet()
	with tf.device('/cpu:0'):
		a = toy_data()
		x = tf.convert_to_tensor(a, np.float32) 
		x_est = fbpcnn(x)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		x_est_np = sess.run(x_est)

	return x_est_np






a = test_fbpconvnet()
print(a)


