from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import utilities


def build_model(num_classes, shape):
	prior = VGG16(							# load VGG16 model
		include_top=False,
		weights='imagenet',
		input_shape=(shape, shape, 3)
	)
	model = Sequential()						# create model
	split_at = -4
	for layer in prior.layers[:split_at]:				# freeze first convolutional blocks
		layer.trainable = False
		model.add(layer)
	for layer in prior.layers[split_at:]:				# fine-tune the last convolutional block
		layer.trainable = True
		model.add(layer)
	model.add(GlobalAveragePooling2D())				# use GlobalAveragePooling2D to reduce overfitting
	model.add(Dense(256, activation='relu', name='Dense_1'))
	model.add(Dropout(0.4, name='Dropout_1'))
	model.add(Dense(512, activation='relu', name='Dense_2'))
	model.add(Dropout(0.2, name='Dropout_2'))
	model.add(Dense(num_classes, activation='softmax', name='Output'))
	return model


def fit_model(generators, model, num_epochs, batch_size, class_weight):
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# define the callbacks
	best_model_checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.1, min_lr=0.000001, verbose=1)
	early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

	history = model.fit_generator(  				# fit model
		generator=generators[0],
		steps_per_epoch=len(generators[0].filenames) // batch_size,
		epochs=num_epochs,
		class_weight=class_weight,
		validation_data=generators[1],
		validation_steps=len(generators[0].filenames) // batch_size,
		callbacks=[best_model_checkpoint, reduce_lr, early_stop],
		verbose=2,
		shuffle=False
	)
	utilities.plot_learning_curves(history)  			# plot diagnostic learning curves

	return model


def load_weights(model, file_name):
	model.load_weights(file_name)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
