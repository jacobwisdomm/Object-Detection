# Import Libraries
import cv2, time, os
import tensorflow as tf
import numpy as np

# Setup to check a file is downloaded for later use
from tensorflow.python.keras.utils.data_utils import get_file

# Generates a seed so the array that is generated later can be filled with the same numbers, instead of manually entering them
np.random.seed(2023)

class Perceptron:
	
	# Nothing is used in this parameter, so set it to self and pass it
	def __init__(self):
		pass
	
	# Reads the class names from the file given in the main program
	def getClasses(self, classFilePath):
	
		# Open file in read mode as a string
		with open(classFilePath, 'r') as f:
			# When the classes are read, split them individually into an array
			self.classList = f.read().splitlines()
			
		# Create a color list between RGB value 0 and 255, and make the list as long as the class list
		self.colorList = np.random.uniform(low = 0, high = 255, size = len(self.classList))
		
		# Print the class list for debugging
		print(len(self.classList), len(self.colorList))
	
	# Downloads the model if it has not been downloaded already
	def downloadModel(self, modelURL):
	
		# Extracts the file name from the model URL
		fileName = os.path.basename(modelURL)
		
		# Extracts the model name from the file name by copying it until it hits the first '.'
		self.modelName = fileName[:fileName.index('.')]
		
		# Name of the cache where the models will be stored (and loaded from)
		self.cacheDir = "./pretrained_models"
		
		# Creates directory, and skips if the file already exists
		os.makedirs(self.cacheDir, exist_ok = True)
		
		# Creates a model file given the file name, the origin of the model, the directory of the cache, and extracts 			# the .tar.gz file
		get_file(fname = fileName, origin = modelURL, cache_dir = self.cacheDir, 
			cache_subdir = "downloaded-models", extract = True)
	
	# Loads the model
	def loadModel(self):
	
		print("Loading Model " + self.modelName)
		
		# Clears memory to improve performance
		tf.keras.backend.clear_session()
		
		# Loads the model from the directory
		self.model = tf.saved_model.load(os.path.join(self.cacheDir, "downloaded-models", self.modelName, "saved_model"))
		print("Model " + self.modelName + " Was Loaded Successfully!")
	
	# Creates bounding boxes based on the image and threshold
	def createBoundingBox(self, image, threshold = 0.5):

		# The image is loaded in BGR format, so convert it to RGB
		inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		
		# Converts to TensorFlow tensor as an 8-bit unsigned integer
		inputTensor = tf.convert_to_tensor(inputTensor, dtype = tf.uint8)
		
		# Expands the array of the image by 1
		inputTensor = inputTensor[tf.newaxis,...]
		
		# Create a variable that handles the detections
		detections = self.model(inputTensor)
		
		# Extracts bounding boxes and converts to numpy array
		bBoxes = detections['detection_boxes'][0].numpy()
		
		# Extracts classes as a numpy array (from text to array to numpy array) 
		classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
		
		# Extracts the confidence scores and converts to a numpy array
		classScores = detections['detection_scores'][0].numpy()
		
		# image height, width, and number of channels
		imageHeight, imageWidth, imageChannels = image.shape
		
		# Applies non max suppression, which limits the amount of boxes to 50, with an overlap based on the threshold
		bBoxIndex = tf.image.non_max_suppression(bBoxes, classScores, max_output_size = 50, iou_threshold = threshold, score_threshold = threshold)
		
		print(bBoxIndex)
		
		# if the length of the index is more than 0
		if len(bBoxIndex) != 0:
			for i in bBoxIndex:
			
				# Creates a listed tuple for the bounding boxes generated 
				bBox = tuple(bBoxes[i].tolist())
				
				# Calculates the class confidence at each iteration of i
				classConfidence = round(100*classScores[i])
				classIndex = classIndexes[i]
				
				# Gets the class label from the index with color, while making the text uppercase
				classLabelText = self.classList[classIndex].upper()
				classColor = self.colorList[classIndex]
				
				# Displays the label and confidence of the item near the bounding box in string format
				displayText = '{}: {}%'.format(classLabelText, classConfidence)
				
				# Values of the pixels at the four corners of the bounding box
				ymin, xmin, ymax, xmax = bBox
				
				# Calculates the actual pixel loacations for xmin, xmax, ymin, and ymax based on the resolution of 					# the image
				xmin, xmax, ymin, ymax = (xmin*imageWidth, xmax*imageWidth, ymin*imageHeight, ymax*imageHeight)
				
				# Converts the pixel locations to integers
				xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
				
				# Draws a rectangle on the image starting at (xmin, ymin) to xmax, ymax)
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = classColor, thickness = 3)
				
				# Displays the class above the bounding box where (0, 0) is the top left of the image
				cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 1, classColor, 2)				
				
		return image
	
	# Predicts the items in an image
	def detectImage(self, imagePath, threshold):
		
		# Loads the image into variable
		image = cv2.imread(imagePath)
		
		# Generates a bounding box based on the image
		bBoxImage = self.createBoundingBox(image, threshold)
		
		# Saves the result picture as a jpeg including the bounding box
		cv2.imwrite(self.modelName + ".jpg", bBoxImage)
		
		# Creates a window named result containing the image with a bounding box
		cv2.imshow("Result", bBoxImage)
		
		# Display the window indefinitely then destroy all windows once the window is closed
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	# Predicts the items of a video
	def detectVideo(self, videoPath, threshold):
	
		# Loads the variable with the video specified in the main function
		video = cv2.VideoCapture(videoPath)

		# If the video fails to open then print an error and return
		if (video.isOpened() == False):
			print("Error Opening file...")
			return
		
		# If the video is loads, load the first frame of the video as an image
		(success, image) = video.read()
		
		# Initializes the start time for FPS
		startTime = 0
		
		while success:
		
			# The current time 
			currentTime = time.time()
			
			# Calculates FPS based on current and start times
			fps = 1/(currentTime - startTime)
			
			# Updates start time to current time (so it doesn't stay on zero)
			startTime = currentTime
			
			# Creates a bounding box based on the image generated from the frame of video
			bBoxImage = self.createBoundingBox(image, threshold)
			
			# Outputs the FPS onto the window
			cv2.putText(bBoxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2)
			
			cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
			cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, 0)

			# Generate a window named "Result" with the image
			cv2.imshow("Result", bBoxImage)
			
			# Wait for a key press to exit the program
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			
			# Reads the next frame as an image if the video is still successfully playing
			(success, image) = video.read()
		
		# Destroy all windows when the loop is broken
		cv2.destroyAllWindows()