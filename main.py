import tensorflow as tf
from tensorflow import keras
import discriminator, generator
import os
import trainer
import time

from matplotlib import pyplot as plt
#from IPython import display
import file_utils as fu
import variables

train_urls, test_urls = fu.get_random_file()
train_dataset, test_dataset = fu.get_datasets(train_urls,test_urls)

#start traininig
trainer.train(train_dataset, test_dataset, variables.epochs, train_urls)

