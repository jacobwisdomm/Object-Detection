#	How to Run on Jetson Nano:
#	1. Open Terminal and type: cd Project
#	2. Copy and Paste: OPENBLAS_CORETYPE=ARMV8 python3
#	3. Type the name of the .py program, in this case Main.py

#	ModelZoo, where pretrained models can be found and tested
#	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md   

# Import the backend program containing Mumpy, TensorFlow, etc...
from HiddenLayers import *

# SSD Mobilenet V2, a lightweight pretrained model
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

# SSD MobileNet V2 but LITE
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

# Name of the file containing the classes the AI can detect
classFile = "coco.names.txt"

# Different things to AI can detect, COMMENT OUT THE PATH THAT IS NOT BEING USED!
#imagePath = "test0.jpg"
videoPath = 0

# NANO VIDEO PATH
#videoPath = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink"

# The amount of overlap the bounding boxes can have if they are the same item (in percent)
threshold = 0.5

# Initialize perceptron class
perceptron = Perceptron()

# Reads classes from the class file
perceptron.getClasses(classFile)

# Downloads (and checks) the program from TensorFlows Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
perceptron.downloadModel(modelURL)

# Loads the model from the directory
perceptron.loadModel()

# The paths an image/video can take, COMMENT OUT THE PATH THAT IS NOT BEING USED!
#perceptron.detectImage(imagePath, threshold)
perceptron.detectVideo(videoPath, threshold)