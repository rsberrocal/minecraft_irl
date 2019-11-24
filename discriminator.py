import tensorflow as tf
import variables
from tensorflow import keras
import generator


def Discriminator():
	ini = keras.layers.Input(shape=[None, None, 3], name="input_img")
	gen = keras.layers.Input(shape=[None, None, 3], name="gener_img")
	
	concat = keras.layers.concatenate([ini, gen])
	
	initializer = tf.random_normal_initializer(0, 0.02)
	
	encode1 = generator.encoder_layer(64, apply_batchnorm=False)(concat)
	encode2 = generator.encoder_layer(128)(encode1)
	encode3 = generator.encoder_layer(256)(encode1)
	encode4 = generator.encoder_layer(512)(encode1)
	encode5 = generator.encoder_layer(1024)(encode1)
	
	last = tf.keras.layers.Conv2D(filters=1,
								  kernel_size=4,
								  strides=1,
								  kernel_initializer=initializer,
								  padding="same"
								  )(encode5)
	return tf.keras.Model(inputs=[ini, gen], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
	loss_object = keras.losses.BinaryCrossentropy(from_logits=True)
	
	# Diferencia entre los true(ya que es la imagen real) y el detectado por el discriminador)
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
	# Diferencia entre los false(ya que es la foto generada y el detectado por el discriminador)
	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
	
	total_disc_loss = real_loss + generated_loss
	
	return total_disc_loss
