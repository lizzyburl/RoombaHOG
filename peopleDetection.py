from svmutil import *
import cv2
import numpy as np
import os
import hog
import time
import roomba

# Takes a window of which to look for a person and resizes it to fit the 64x128 size
def grabImage(img, startRow, startCol, endRow, endCol):
	resizeIm = img[startRow:endRow, startCol:endCol]
	#cv2.imshow("resizeIm", resizeIm)
	#cv2.waitKey(0)
	return cv2.resize(resizeIm, (64,128))

offset = 20
#roombaBot = roomba.Roomba()

# Load the SVM model that was pre-computed
m = svm_load_model('hog_people.model')

# Open up the webcam
cap = cv2.VideoCapture(1)
num =0
# Give me time to run in front of the camera to test it
time.sleep(.5)
while (True):
	startTime = time.time()
	# Capture frame-by-frame
	ret, frame = cap.read()

	#gTemp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("tempFrame.png", frame)
	gray = np.double(cv2.imread('tempFrame.png', cv2.CV_LOAD_IMAGE_GRAYSCALE))/255
	# Display the resulting frame
	cv2.imshow('frame',gray)
	cv2.waitKey(1)

	# This is an array of the total row lengths that we will be looking at. 
	# All images have 2:1 row:col ratio
	# These will be scaled into 64x128 images in grabImage
	scales = [480, 440, 400]
	scaleIndices = [0, 21, 87]

	# x will contain a list of dictionaries that we give the SVM. 
	# y contains a list of labels. I just set all labels to 1, because this will give us the 
	# number of people that are spotted.
	x = []
	y = []
	i = 0
	for scale in scales:
		overlappingRows = (480 - scale)/offset + 1
		overlappingCols = (640 - scale/2)/offset + 1
		for j in range(0, overlappingRows*overlappingCols):
			dictionary = {}

			# Figure out the square that we want to look at in the context of j, the index segments 
			# for that scale
			startRow = int(np.floor(j/overlappingCols)*offset + 1)
			endRow = int(startRow + scale - 1) 
			startCol = int(j - np.floor(j/overlappingCols)*overlappingCols)*offset + 1
			endCol = int(startCol + scale/2 ) -1

			# This needs to be in order to add the rectangle without writing over the image and having 
			# numerous rectangles
			grayCopy = gray.copy()
			cv2.rectangle(grayCopy, (startCol, startRow), (endCol, endRow), (57,173,52))
			cv2.imshow("frame", grayCopy)
			cv2.waitKey(1)

			# Grab the 64x128 section we want to look at for people
			grayRightSize = grabImage(gray, startRow, startCol, endRow, endCol)
			# Get the HOG vector!
			v = hog.getImageVector(grayRightSize)

			# The counter serves as a feature label for the dimension
			counter = 1

			# Just set the "true" label. As I mentioned, this doesn't really matter.
			y.append(1)
			# Create the dictionary for x
			for dim in v:
				if dim != 0:
					dictionary[counter] = dim
				counter = counter + 1;
			x.append(dictionary)
			i = i + 1

	p_label, p_acc, p_val = svm_predict(y, x, m);

	# p_val will be a greater positive number if there is a person and a lower negative number
	# if there is not a person
	maxIndex = np.argmax(p_val)
	print maxIndex

	print p_val[int(maxIndex)]


	# Figure out what scale the preferred image was at.
	chosenScale = 0
	for i in range(0, len(scales)):
		if maxIndex >= scaleIndices[i]:
			chosenScale = i
			

	# Print the index it was in that scale, for debugging purposes
	print scaleIndices
	scale = scales[chosenScale]
	scaleIndex = maxIndex - scaleIndices[chosenScale]


	print scale
	print "scale Index: " + str(scaleIndex)

	# Redoing the math we did before to find the square with the best fit.
	overlappingCols = (640 - scale/2)/offset + 1
	startRow = int(np.floor(scaleIndex/overlappingCols)*offset + 1)
	endRow = int(startRow + scale - 1) 
	startCol = int(scaleIndex - np.floor(scaleIndex/overlappingCols)*overlappingCols)*offset + 1
	endCol = int(startCol + scale/2 ) -1
	print startRow, endRow, startCol, endCol
	cv2.rectangle(gray, (startCol, startRow), (endCol, endRow), (57,173,52))
	cv2.imshow('closestThingToPerson', gray)
	if p_val[int(maxIndex)][0] > 0:
		cv2.imwrite(('person'+str(num)+'prob'+str(p_val[int(maxIndex)][0])+'.png'),gray*255)
		print "We are pretty sure there is a person"
		# Time to move the roomba! 
		# If the person is on the left, turn left
		# if startCol < 150:
		# 	roombaBot.turn_left();
		# 	time.sleep(3);
		# 	roombaBot.stop();
		# If the person is on the right, turn right 
		# if startCol > 350:
		# 	roombaBot.turn_right();
		# 	time.sleep(3);
		# 	roombaBot.stop();
		# If the person is at the smallest scale, we should move forward.
		# if scale==400:
		# 	roombaBot.move_forward();
		# 	time.sleet(3);
		# 	roombaBot.stop();
		# If the person is at the largest scale, we should move backwards
		# if scale==480:
		# 	roombaBot.move_backward();
		# 	time.sleep(3);
		# 	roombaBot.stop();
	else:
		# If we can't find a person, we don't really know what to do.
		# Let's just turn 
		# roombaBot.turn_left_in_place();
		# time.sleep(4);
		# roombaBot.stop();
		cv2.imwrite('notPerson'+str(num)+'prob'+str(p_val[int(maxIndex)][0])+'.png',gray*255)
		print "No person here!"
	endTime = time.time() - startTime
	print "Time Elapsed: " + str(endTime)
	num = num + 1

