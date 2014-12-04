from svmutil import *
import cv2
import numpy as np
import os
import hog

# prepareOutput: prepares the testing and training output in a format that is usable for libsvm
# Parameters:
#	outputFileName: the name of the output file (such as "hogTrainingOutput")
#	positivePath: a string containing the path to a folder of images containing people
#	negativePath: a string containg the path to a folder of images with out people
# Notes: We assumed that all training images were of size 64x128, and it was treated as such.
def prepareOutput(outputFileName, positivePath, negativePath):
	output = open(outputFileName , 'wb');

	# Go through and mark the positive files for the SVM
	for filename in os.listdir(positivePath):
		pathPlusFilename = os.path.join(positivePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))

		# Get the 3780-d vector that contains the concatenated histograms for each bin
		v = hog.getImageVector(img)
		# We are using a counter to give each "feature" a label
		counter = 1
		# Mark that it is a posive path
		output.write("+1 ")
		# Write each dimension in the format that we need.
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")
	print "Done with positive"
	for filename in os.listdir(negativePath):
		pathPlusFilename = os.path.join(negativePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
		v = hog.getImageVector(img)
		counter = 1
		output.write("-1 ")
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")
	output.close()
	print "Done with negative"
#prepareOutput('hogTraining', "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\smallerPed", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\smallerNotPed")
#prepareOutput("hogTesting", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\testPeople", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\testNotPed")

y, x = svm_read_problem('hogTraining');
m = svm_train(y, x, '-t 0 -s 0');
svm_save_model('hog_people.model', m)

yTest, xTest = svm_read_problem('hogTesting');

p_label, p_acc, p_val = svm_predict(yTest, xTest, m);
#print p_val
