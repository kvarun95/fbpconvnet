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
		

		# Common layers
		self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))
		self.upsamp = tf.keras.layers.UpSampling2D(size=(2,2))
		self.conc = tf.keras.layers.Concatenate(axis=3)
		# self.bnorm = tf.keras.layers.BatchNormalization()

		# Phase I - Analysis steps-downward process
		# slab 1
		self.conv1a = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1b = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# slab 2
		self.conv2a = tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv2b = tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		# self.pool2 = tf.keras.layers.maxPool2D(pool_size=(2,2))


		# Phase II - Synthesis steps-upward process
		# slab 2
		self.convu2 = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1c = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv1d = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')
		self.conv_final = tf.keras.layers.Conv2D(1, (1,1), activation=tf.nn.relu, kernel_initializer=initializer, padding='same')

		# skip connection
		self.skipc = tf.keras.layers.Add()


	def call(self, x, training=None):


		# Phase I - Analysis step-downward process
		# slab 1
		x1 = self.conv1a(x)
		c1 = self.conv1b(x1)
		x1 = self.pool(c1)

		# slab 2
		x2 = self.conv2a(x1)
		x2 = self.conv2b(x2)

		# slab 1
		x1 = self.convu2(self.upsamp(x2))
		x1 = self.conv1c(self.conc([x1, c1]))
		x1 = self.conv1d(x1)

		x1 = self.conv_final(x1)

		# skip connection
		x_est = self.skipc([x1, x])

		return x_est



















		








		












	pass





























