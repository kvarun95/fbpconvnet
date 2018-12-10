import os
import numpy as np 
import tensorflow as tf 

''' This file defines the computational graph and 
'''
class fbpconvnet(tf.keras.Model):

	''' Attributes
	

	'''

	def __init__(self):

		# Initialize superclass
		super().__init__()

		# initialize weights
		initializer = tf.variance_scaling_initializer(scale=2.0)
		# initializer = tf.glorot_uniform_initializer()

		# Common layers
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
		self.upsamp1 = tf.keras.layers.UpSampling2D(size=(2,2))
		self.upsamp2 = tf.keras.layers.UpSampling2D(size=(2,2))
		self.conc1 = tf.keras.layers.Concatenate(axis=3)
		self.conc2 = tf.keras.layers.Concatenate(axis=3)
		self.bnorm1 = tf.keras.layers.BatchNormalization()
		self.bnorm2 = tf.keras.layers.BatchNormalization()
		self.bnorm3 = tf.keras.layers.BatchNormalization()

		# Phase I - Analysis steps-downward process
		# slab 1
		self.conv1a = tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1b = tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# slab 2
		self.conv2a = tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv2b = tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# slab 3
		self.conv3a = tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv3b = tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# Phase II - Synthesis steps-upward process
		# slab 2
		self.convu3 = tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv2c = tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv2d = tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		
		# slab 1
		self.convu2 = tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1c = tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1d = tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv_final = tf.keras.layers.Conv2D(1, (1,1), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# skip connection
		self.skipc = tf.keras.layers.Add()


	def call(self, x):


		if True:
			# Phase I - Analysis step-downward process
			# slab 1
		# 	x1 = self.bnorm1(self.conv1a(x)) 
		# 	c1 = self.bnorm1(self.conv1b(x1))
		# 	x1 = self.pool1(c1)

		# 	# slab 2
		# 	x2 = self.bnorm2(self.conv2a(x1)) 
		# 	c2 = self.bnorm2(self.conv2b(x2)) 
		# 	x2 = self.pool2(c2)

		# 	# slab 3
		# 	x3 = self.bnorm3(self.conv3a(x2)) 
		# 	x3 = self.bnorm3(self.conv3b(x3)) 

		# 	# slab 2
		# 	x2 = self.bnorm2(self.convu3(self.upsamp2(x3))) 
		# 	# x2 = self.bnorm2(self.conv2c(self.conc2([x2, c2]))) 
		# 	x2 = self.bnorm2(self.conv2c(x2)) 
		# 	x2 = self.bnorm2(self.conv2d(x2)) 

		# 	# slab 1
		# 	x1 = self.bnorm1(self.convu2(self.upsamp1(x2))) 
		# 	# x1 = self.bnorm1(self.conv1c(self.conc1([x1, c1]))) 
		# 	x1 = self.bnorm1(self.conv1c(x1)) 
		# 	x1 = self.bnorm1(self.conv1d(x1)) 

		# 	x1 = self.conv_final(x1)

		# 	# skip connection
		# 	# x_est = self.skipc([x1, x])
		# 	x_est = x1

		# else:
			# Phase I - Analysis step-downward process
			# slab 1
			x1 = self.conv1a(x) 
			c1 = self.conv1b(x1)
			x1 = self.pool1(c1)

			# slab 2
			x2 = self.conv2a(x1)
			c2 = self.conv2b(x2)
			x2 = self.pool2(c2)

			# slab 3
			x3 = self.conv3a(x2) 
			x3 = self.conv3b(x3)

			# slab 2
			x2 = self.convu3(self.upsamp2(x3))
			x2 = self.conv2c(self.conc2([x2, 0.0001*c2]))
			# x2 = self.conv2c(x2)
			x2 = self.conv2d(x2)

			# slab 1
			x1 = self.convu2(self.upsamp1(x2))
			x1 = self.conv1c(self.conc1([x1, 0.0001*c1]))
			# x1 = self.conv1c(x1) 
			x1 = self.conv1d(x1)

			x1 = self.conv_final(x1)

			# skip connection
			x_est = self.skipc([x1, 0.00000*x])
			# x_est = x1

		return x_est



















		








		












	pass





























