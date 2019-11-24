import file_utils as fu
import tensorflow as tf
#import variables
train_urls, test_urls = fu.get_random_file()
train_dataset, test_dataset = fu.get_datasets(train_urls,test_urls)

#start traininig
import trainer as tr

tr.train(train_dataset, test_dataset, 2, train_urls)

