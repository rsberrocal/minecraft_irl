import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import variables


def load(image_file):
	image = tf.io.read_file(image_file)
	image = tf.image.decode_jpeg(image)
	
	w = tf.shape(image)[1]
	
	w = w // 2
	real_image = image[:, :w, :]
	input_image = image[:, w:, :]
	
	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)
	
	return input_image, real_image


def loadAllFiles():
	files_list = []
	for root, directory, file in os.walk(variables.INPUT_DIR):
		files_list.append(os.path.join(root, file))
	return files_list


def getRandomFiles():
	rand_urls = np.copy(loadAllFiles())
	np.random.shuffle(rand_urls)
	
	train_urls = rand_urls[:variables.train_n]
	test_urls = rand_urls[variables.train_n:variables.n]
	
	return train_urls, test_urls


# Funcion que redimensionara la imagen a los valores que tiene el programa
# Los valores se encuentran en variables.py
def resize(input_image, real_image):
	input_image = tf.image.resize(input_image, [variables.HEIGHT, variables.WIDTH],
								  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	real_image = tf.image.resize(real_image, [variables.HEIGHT, variables.WIDTH],
								 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return input_image, real_image


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
	input_image = (input_image / ((variables.HEIGHT // 2) - 0.5)) - 1
	real_image = (real_image / ((variables.HEIGHT // 2) - 0.5)) - 1
	
	return input_image, real_image
