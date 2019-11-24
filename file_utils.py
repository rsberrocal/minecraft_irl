import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import variables


def load_image_file(image_file):
	image = tf.io.read_file(image_file)
	image = tf.image.decode_jpeg(image)
	
	w = tf.shape(image)[1]
	
	w = w // 2
	real_image = image[:, :w, :]
	input_image = image[:, w:, :]
	
	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)
	
	return input_image, real_image


def load_all_files():
	files_list = []
	for root, directory, file in os.walk(variables.INPUT_DIR):
		files_list = file
	return files_list


def get_random_file():
	rand_urls = np.copy(load_all_files())
	np.random.shuffle(rand_urls)
	
	train_urls = rand_urls[:variables.train_n]
	test_urls = rand_urls[variables.train_n:variables.n]
	return train_urls, test_urls


# Funcion que redimensionara la imagen a los valores que tiene el programa
# Los valores se encuentran en variables.py
def resize(input_image, real_image, height, width):
	input_image = tf.image.resize(input_image, [height, width],
								  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	real_image = tf.image.resize(real_image, [height, width],
								 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return input_image, real_image


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
	input_image = (input_image / ((variables.HEIGHT // 2) - 0.5)) - 1
	real_image = (real_image / ((variables.HEIGHT // 2) - 0.5)) - 1
	
	return input_image, real_image


def random_crop(input_image, real_image):
	# Para hacer el crop se apilan ambas imagenes y se recorta de forma aleatoria
	stacked_image = tf.stack([input_image, real_image], axis=0)
	cropped_image = tf.image.random_crop(
		stacked_image, size=[2, variables.HEIGHT, variables.WIDTH, 3])
	
	return cropped_image[0], cropped_image[1]


# Aumentar datos Modifica una sola imagen para generar mas imagenes
# @tf.function()
def random_jitter(input_image, real_image):
	# resizing to 286 x 286 x 3
	input_image, real_image = resize(input_image, real_image, variables.HEIGHT + 50, variables.WIDTH + 50)
	
	# randomly cropping to 256 x 256 x 3
	input_image, real_image = random_crop(input_image, real_image)
	
	if tf.random.uniform(()) > 0.5:
		# random mirroring
		input_image = tf.image.flip_left_right(input_image)
		real_image = tf.image.flip_left_right(real_image)
	
	return input_image, real_image


def get_real_path(path):
	index_dot = path.rfind('.')
	index_slash = path.rfind('/')
	name = path[index_slash:index_dot]
	ext = path[index_dot:]
	name = name[:-1]
	name = name + '1'
	return variables.OUTPUT_DIR + name + ext

@tf.function
def load_image(file, augment=True):
	inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(variables.INPUT_DIR + '/' + file)), tf.float32)[..., :3]
	reimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(variables.OUTPUT_DIR + '/' + file)), tf.float32)[..., :3]
	
	inimg, reimg = resize(inimg, reimg, variables.HEIGHT, variables.WIDTH)
	if augment:
		inimg, reimg = random_jitter(inimg, reimg)
	inimg, reimg = normalize(inimg, reimg)
	
	return inimg, reimg


def load_image_train(file):
	return load_image(file)


def load_image_test(file):
	return load_image(file, False)


def get_datasets(tr_urls, ts_urls):
	train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
	train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	train_dataset = train_dataset.batch(1)
	
	test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
	test_dataset = test_dataset.map(load_image_test)
	test_dataset = test_dataset.batch(1)
	return train_dataset, test_dataset
