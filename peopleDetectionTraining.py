from svmutil import *
import cv2
import numpy as np
import os
import hogStuff


def prepareOutput(outputFileName, positivePath, negativePath):
	output = open(outputFileName , 'wb');
	i = 1
	for filename in os.listdir(positivePath):
		pathPlusFilename = os.path.join(positivePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))/255
		# print filename
		v = hogStuff.getImageVector(img)
		counter = 1
		output.write("+1 ")
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")
		i = i +1
	print i
	i = 1
	for filename in os.listdir(negativePath):
		pathPlusFilename = os.path.join(negativePath,filename)
		img = np.double(cv2.imread(pathPlusFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE))/255
		# print filename
		v = hogStuff.getImageVector(img)
		counter = 1
		output.write("-1 ")
		for dim in v:
			if dim != 0:
				output.write(str(counter) + ":" + str(dim) + " ")
			counter = counter + 1;
		output.write("\n")
		i = i+1
	print i
	output.close()

prepareOutput('hogTrainingOutput', "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\pedestrians128x64", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\notPedestrians128x64")
prepareOutput("test", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\testPeople", "C:\\Users\\Lizzy\\Documents\\ComputerVision\\FinalProject\\testNotPed")

y, x = svm_read_problem('hogTrainingOutput');
print len(y)
print y[1]
print len(x)
print x[1][1]
m = svm_train(y, x, '-t 0 -s 0');
svm_save_model('hog_people.model', m)

yTest, xTest = svm_read_problem('test');
p_label, p_acc, p_val = svm_predict(yTest, xTest, m);

print p_acc;
