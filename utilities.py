import cv2
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def load_data(folder_path, batch_size, shape, classes):
	datagenerator = ImageDataGenerator(							# load and augment data
		rescale=1. / 255,
		rotation_range=20,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest',
		validation_split=0.2
	)
	train_generator = datagenerator.flow_from_directory(
		directory=folder_path + '/train',
		classes=classes,
		target_size=(shape, shape),
		batch_size=batch_size,
		class_mode='categorical',
		subset='training'
	)
	val_generator = datagenerator.flow_from_directory(
		directory=folder_path + '/train',
		classes=classes,
		target_size=(shape, shape),
		batch_size=batch_size,
		class_mode='categorical',
		subset='validation'
	)
	return train_generator, val_generator


def load_classes(folder_path):
	return [c for c in os.listdir(folder_path + '/train')]


def weight_classes(folder_path, classes):
	class_count = {}													# count the number of images per class
	count = 0
	for i in range(len(classes)):
		for _ in glob.glob(folder_path + '/train/' + classes[i] + '/*.jpg'):
			count += 1
		class_count[i] = count
		count = 0
	total_count = sum(class_count.values())
	return {cls: total_count / count for cls, count in enumerate(class_count.values())}


def load_test(folder_path, shape):
	testX = []  														# load test images
	for img in glob.glob(folder_path + '/test/*.jpg'):
		img = cv2.imread(img)
		img = cv2.resize(img, (shape, shape))
		img = image.img_to_array(img)
		testX.append(img)
	testX = np.array(testX, dtype='float') / 255.0
	return testX


def output_predictions(predictions, classes, testX):
	submission = pd.DataFrame({
		'id': ['image_{:04d}'.format(i) for i in range(len(testX))],
		'label': [classes[np.argmax(pred)] for pred in predictions]
	})
	submission.to_csv('submission.csv', index=False)


def plot_learning_curves(history):
	plt.figure(figsize=[8, 6])											# loss curve
	plt.plot(history.history['loss'], 'b', linewidth=3.0)
	plt.plot(history.history['val_loss'], 'r', linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
	plt.xlabel('Epochs ', fontsize=16)
	plt.ylabel('Loss', fontsize=16)
	plt.title('Loss Curves', fontsize=16)
	plt.savefig('loss_plot.png')
	plt.show()
	plt.figure(figsize=[8, 6])											# accuracy curve
	plt.plot(history.history['accuracy'], 'b', linewidth=3.0)
	plt.plot(history.history['val_accuracy'], 'r', linewidth=3.0)
	plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
	plt.xlabel('Epochs ', fontsize=16)
	plt.ylabel('Accuracy', fontsize=16)
	plt.title('Accuracy Curves', fontsize=16)
	plt.savefig('acc_plot.png')
	plt.show()
