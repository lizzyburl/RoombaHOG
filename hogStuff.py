import numpy as np
import cv2
import slitUpBlocks
from numpy import linalg as LA


def getImageVector(img):
	height, width = img.shape

	#cv2.imshow('image',img);
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	# Run each image through the mask in the x and y direction
	# filter2d [-1 0 1]
	kern = np.matrix([-1, 0, 1]);

	kernTrans = np.transpose(kern);
	gx = cv2.filter2D(img, -1, kern)
	gy = cv2.filter2D(img, -1, kernTrans)

	# Compute gradient fro magnitude and orientation
	# M(x,y) = sqrt(gx^2 + gy^2)
	magnitude = np.sqrt(np.multiply(gx,gx) + np.multiply(gy,gy));
	#cv2.imshow('image',magnitude);
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	# O(x,y) = tan^-1(gy/gx)
	orientation = np.arctan2(gy, gx);
	#cv2.imshow('image', orientation);
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	v = slitUpBlocks.overlappingBlocksHistogram(magnitude, orientation)

	vNorm = np.abs(LA.norm(v));
	v = v/vNorm
	return v

