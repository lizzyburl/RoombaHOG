from svmutil import *
import cv2
import numpy as np
import os
import hogStuff
import time

def grabImage(img, startRow, startCol, endRow, endCol):
	resizeIm = img[startRow:endRow, startCol:endCol]
	return cv2.resize(resizeIm, (64,128))

m = svm_load_model('hog_people.model')

cap = cv2.VideoCapture(1)
for i in range(0,5):
	ret, frame = cap.read()
time.sleep(5)
while (True):

	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# This is an array of the total row lengths that we will be looking at. 
	# All images have 2:1 row:col ratio
	scales = [480, 440, 400]
	scaleIndices = []
	x = []
	y = []

	i = 0
	for scale in scales:
		scaleIndices.append(i)
		overlappingRows = (480 - scale)/20 + 1
		overlappingCols = (640 - scale/2)/20 + 1
		dictionary = {}
		for j in range(0, overlappingRows*overlappingCols):
			startRow = int(np.floor(j/overlappingCols)*20 + 1)
			endRow = int(startRow + scale - 1) 
			startCol = int(j - np.floor(j/overlappingCols)*overlappingCols)*20 + 1
			endCol = int(startCol + scale/2 ) -1
			grayCopy = gray.copy()
			cv2.rectangle(grayCopy, (startCol, startRow), (endCol, endRow), (57,173,52))
			cv2.imshow("frame2", grayCopy)
			grayRightSize = grabImage(gray, startRow, startCol, endRow, endCol)
			v = hogStuff.getImageVector(grayRightSize)
			counter = 1
			y.append(1)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			#time.sleep(.5)
			for dim in v:
				if dim != 0:
					dictionary[counter] = dim
				counter = counter + 1;
			x.append(dictionary)
			i = i + 1


	p_label, p_acc, p_val = svm_predict(y, x, m);

	maxIndex = np.argmax(p_val)
	print maxIndex

	print p_val[int(maxIndex)]



	chosenScale = 0
	for i in range(0, len(scales)):
		if maxIndex >= scaleIndices[i]:
			chosenScale = i
			break

	print scaleIndices
	scale = scales[chosenScale]
	scaleIndex = maxIndex - scaleIndices[chosenScale]
	print scale
	print "scale Index: " + str(scaleIndex)
	overlappingCols = (640 - scale/2)/20 + 1
	startRow = int(np.floor(j/overlappingCols)*20 + 1)
	endRow = int(startRow + scale - 1) 
	startCol = int(j - np.floor(j/overlappingCols)*overlappingCols)*20 + 1
	endCol = int(startCol + scale/2 ) -1
	print startRow, endRow, startCol, endCol
	resizeIm = gray[startRow:endRow, startCol:endCol]
	if p_val[int(maxIndex)][0] > .70:
		cv2.imwrite('bestim.png',resizeIm)

		#cv2.rectangle(gray, (startRow, startCol), (endRow, endCol), (57,173,52))
		#cv2.waitKey(0)

		print "We are pretty sure"
	else:
		cv2.imwrite('badim.png',resizeIm)

