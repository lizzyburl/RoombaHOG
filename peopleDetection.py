from svmutil import *
import cv2
import numpy as np
import os
import hogStuff
import time

def grabCenter(indexOfScale, scale, img, totalRows, totalCols):
	startRow = int(np.floor(indexOfScale/totalCols)*20 + 1)
	endRow = int(startRow + scale - 1) 
	startCol = int(np.floor(indexOfScale/totalRows))*20 + 1
	endCol = int(startCol + scale/2 ) -1
	print startRow, endRow, startCol, endCol
	resizeIm = img[startRow:endRow, startCol:endCol]
	return cv2.resize(resizeIm, (64,128))

m = svm_load_model('hog_people.model')

cap = cv2.VideoCapture(1)

while (True):

	output =  open('webcamOuput' , 'wb');
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	cv2.imshow('frame',gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	scales = [480, 440, 400]
	i = 0
	for scale in scales:
		overlappingRows = (480 - scale)/20 + 1
		overlappingCols = (640 - scale/2)/20 + 1

		for j in range(0, overlappingRows*overlappingCols):
			grayRightSize = grabCenter(j, scale, gray, overlappingRows, overlappingCols)
			v = hogStuff.getImageVector(grayRightSize)
			counter = 1
			output.write("+1 ")
			for dim in v:
				if dim != 0:
					output.write(str(counter) + ":" + str(dim) + " ")
				counter = counter + 1;
			output.write("\n")
			j = j + 1
		i = i + 1
	output.close()

	yTest, xTest = svm_read_problem('webcamOuput');
	p_label, p_acc, p_val = svm_predict(yTest, xTest, m);
	print p_val
