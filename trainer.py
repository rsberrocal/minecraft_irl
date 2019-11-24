import generator, discriminator, dataset
from matplotlib import pyplot as plt
from file_utils import get_datasets as datas


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


def train_step(input_image, target):
	pass


def train(dataset, epochs):
	for epoch in range(epochs):
		imgi = 0
		for input_image, target in dataset:
			print('epoch ' + str(epoch) + ' - train: ' + str(imgi) + '/' + len(tr))


def get_dataset():
	return []
