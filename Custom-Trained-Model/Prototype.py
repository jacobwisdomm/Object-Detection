import cv2, time, os
import tensorflow as tf
import numpy as np


class Perceptron:

# Nothing is used in this parameter, so set it to self and pass it
	def __init__(self):
		pass
	
	# Reads the class names from the file given in the main program
	def modelManip(self, classFilePath, imagePath):
	
		# Open file in read mode as a string
		with open(classFilePath, 'r') as f:
			# When the classes are read, split them individually into an array
			self.classList = f.read().splitlines()
			
		# Create a color list between RGB value 0 and 255, and make the list as long as the class list
		self.colorList = np.random.uniform(low = 0, high = 255, size = len(self.classList))
		
		# Print the class list for debugging
		print(len(self.classList), len(self.colorList))

		print("Loading Data...")
		cifar10 = tf.keras.datasets.cifar10
		(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

		print(train_images.shape)

		train_images = train_images.astype('float32')
		test_images = test_images.astype('float32')
		train_images /= 255
		test_images /= 255

		print("Creating Model...")
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.InputLayer(input_shape = (train_images[0].shape)))
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu')) 
		#model.add(tf.keras.layers.Dense(128, activation = 'relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2))) 
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

		model.compile(optimizer = 'adam',
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
		metrics = ['Accuracy'])

		print("Training the model...")
		model.fit(train_images, train_labels, epochs = 5)

		test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
		print('\nTest Accuracy;', test_acc * 100)


		# testing images begins
		print("Inserting Testing Images...")
		probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
		predictions = probability_model.predict(test_images)

		print("Outputting Results for Image Number 1...")
		print(np.argmax(predictions[0]))
		print(test_labels[0])


		print("Inserting test.jpeg...")
		image = cv2.imread('test.jpeg', 1)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (32, 32))
		image = np.expand_dims(image, 0)
		print(image.shape)
		image_prediction = probability_model.predict(image)

		print("Class Number Detected:", np.argmax(image_prediction[0]))
		#image_predictions = probability_model.predict(image)
		#print(np.argmax(image_predictions))
		#cv2.imshow("Result", image)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
