from svmutil import *
import cv2
import numpy as np
import os
import hogStuff
import time

def grabCenter(img):
	#Center is at 320
	resizeIm = img[1:480, 200:440]
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

	grayRightSize = grabCenter(gray)
	cv2.imwrite('grayRightSize.png', gray)

	v = hogStuff.getImageVector(grayRightSize)
	counter = 1
	output.write("+1 ")
	for dim in v:
		if dim != 0:
			output.write(str(counter) + ":" + str(dim) + " ")
		counter = counter + 1;

	output.close()

	yTest, xTest = svm_read_problem('webcamOuput');
	p_label, p_acc, p_val = svm_predict(yTest, xTest, m);
	if p_label > 0:
		print p_label
	time.sleep(3)
