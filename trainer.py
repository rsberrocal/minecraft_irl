import generator
import discriminator

import variables

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import time


def generate_images(model, test_input, tar):
	prediction = model(test_input, training=True)
	plt.figure(figsize=(15, 15))
	
	display_list = [test_input[0], tar[0], prediction[0]]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']
	
	for i in range(3):
		plt.subplot(1, 3, i + 1)
		plt.title(title[i])
		# getting the pixel values between [0, 1] to plot it.
		plt.imshow(display_list[i] * 0.5 + 0.5)
		plt.axis('off')
	plt.show()


def train_step(input_image, target, gen, gen_opti, discr, discr_opti, summary, epoch):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
		input_aux = input_image
		input_image = np.expand_dims(input_image, axis=0)
		output_image = gen(((input_aux + 1) * 255), training=True)
		
		input_image = np.squeeze(input_image, axis=0)
		output_gen_discr = discr([input_image, output_image], training=True)
		
		output_target_discr = discr([target, input_image], training=True)
		
		discr_loss = discriminator.discriminator_loss(output_target_discr, output_gen_discr)
		
		gen_loss, gan_loss, l1_loss = generator.generator_loss(output_gen_discr, output_image, target)
	
	generator_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
	
	discriminator_grads = discr_tape.gradient(discr_loss, discr.trainable_variables)
	
	gen_opti.apply_gradients(zip(generator_grads, gen.trainable_variables))
	
	discr_opti.apply_gradients(zip(discriminator_grads, discr.trainable_variables))
	
	with summary.as_default():
		tf.summary.scalar('gen_total_loss', gen_loss, step=epoch)
		tf.summary.scalar('gen_gan_loss', gan_loss, step=epoch)
		tf.summary.scalar('gen_l1_loss', l1_loss, step=epoch)
		tf.summary.scalar('disc_loss', discr_loss, step=epoch)


def train(train_dataset, test_dataset, epochs, tr_urls):
	gen = generator.Generator()
	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discr = discriminator.Discriminator()
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	checkpoint = tf.train.Checkpoint(
		generator_optimizer=generator_optimizer,
		discriminator_optimizer=discriminator_optimizer,
		generator=gen,
		discriminator=discr
	)
	
	summary_writer = tf.summary.create_file_writer(
		variables.LOG_DIR + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	
	for epoch in range(epochs):
		start = time.time()
		
		imgi = 0
		for input_image, target in train_dataset:
			print('epoch ' + str(epoch) + ' - train: ' + str(imgi) + '/' + str(len(tr_urls)))
			imgi += 1
			train_step(input_image, target, gen, generator_optimizer, discr, discriminator_optimizer, summary_writer,
					   epoch)
		
		if (epochs + 1) % 3 == 0:
			checkpoint.save(file_prefix=variables.CHECK_DIR + 'ckpt')
		
		print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
	
	checkpoint.save(file_prefix=variables.CHECK_DIR + 'ckpt')
	
	checkpoint.restore(tf.train.latest_checkpoint(variables.CHECK_DIR))
	
	# clear
	imgi = 0
	for inp, tar in test_dataset.take(5):
		generate_images(gen, inp, tar)
		imgi += 1
