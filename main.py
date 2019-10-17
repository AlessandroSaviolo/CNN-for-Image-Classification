import utilities
import network
import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)      			# suppress messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

	''' 0. Set Hyperparameters '''

	num_epochs = 100
	batch_size = 64
	shape = 100
	folder_path = 'cs-ioc5008-hw1'

	''' 1. Load and Preprocess data '''

	classes = utilities.load_classes(folder_path)					# load classes
	generators = utilities.load_data(folder_path, batch_size, shape, classes)	# load and augment data
	testX = utilities.load_test(folder_path, shape)					# load test data

	class_weight = utilities.weight_classes(folder_path, classes)		# weight classes so that we can later provide
																		# bias to minority classes during training

	''' 2. Build and Fit the model '''

	model = network.build_model(len(classes), shape)			# build model using transfer learning

	model = network.fit_model(generators, model, num_epochs, batch_size, class_weight)	# fit model

	''' 3. Restore best model and output predictions '''

	model = network.load_weights(model, 'weights.best.hdf5')		# load best model found

	predictions = model.predict(testX)					# compute output predictions for Kaggle challenge
	utilities.output_predictions(predictions, classes, testX)
