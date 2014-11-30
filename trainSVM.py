import cv2
import numpy as np
import os
import hogStuff

def prepareOutput(outputFileName):
	positivePath = "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\pedestrians128x64"
	negativePath = "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\notPedestrians128x64"
	output = open(outputFileName , 'wb');

	for filename in os.listdir(positivePath):
		pathPlusFilename = os.path.join(positivePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))/255
		print filename
		v = hogStuff.getImageVector(img)
		counter = 1
		output.write("+1 ")
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")

	for filename in os.listdir(negativePath):
		pathPlusFilename = os.path.join(negativePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))/255
		print filename
		v = hogStuff.getImageVector(img)
		counter = 1
		output.write("-1 ")
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")

prepareOutput("hogTrainingOutput")