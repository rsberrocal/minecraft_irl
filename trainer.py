import generator, discriminator
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def generate_images(model, test_input, tar, display_img=True):
	prediction = model(test_input, training=True)
	plt.figure(figsize=(15, 15))
	
	display_list = [test_input[0], tar[0], prediction[0]]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']
	
	if display_img:
		for i in range(3):
			plt.subplot(1, 3, i + 1)
			plt.title(title[i])
			# getting the pixel values between [0, 1] to plot it.
			plt.imshow(display_list[i] * 0.5 + 0.5)
			plt.axis('off')
		plt.show()


def train_step(input_image, target, gen, discr):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
		input_image = np.expand_dims(input_image, axis=0)
		output_image = gen(((input_image + 1) * 255), training=True)
		
		input_image = np.squeeze(input_image, axis=0)
		output_gen_discr = discr([input_image, output_image], training=True)
		
		output_target_discr = discr([target, input_image], training=True)
		
		discr_loss = discriminator.discriminator_loss(output_target_discr, output_gen_discr)
		
		gen_loss = generator.generator_loss(output_gen_discr, output_image, target)
		
		generator_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
		
		discriminator_grads = discr_tape.gradient(discr_loss, discr.trainable_variables)
		
		generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		
		generator_optimizer.apply_gradients(zip(generator_grads, gen.trainable_variables))
		
		discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		
		discriminator_optimizer.apply_gradients(zip(discriminator_grads, discr.trainable_variables))


def train(train_dataset, test_dataset, epochs, tr_urls):
	gen = generator.Generator()
	discr = discriminator.Discriminator()
	for epoch in range(epochs):
		imgi = 0
		for input_image, target in train_dataset:
			print('epoch ' + str(epoch) + ' - train: ' + str(imgi) + '/' + str(len(tr_urls)))
			imgi += 1
			train_step(input_image, target, gen, discr)
	# clear
	for inp, tar in test_dataset.take(5):
		generate_images(generator, inp, tar, str(imgi) + '_' + str(epoch), )
