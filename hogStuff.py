import numpy as np
import cv2
import slitUpBlocks
from numpy import linalg as LA

# 
def getImageVector(img):
	height, width = img.shape

	# Run each image through the mask in the x and y direction
	# filter2d [-1 0 1]
	kern = np.matrix([-1, 0, 1]);

	kernTrans = np.transpose(kern);
	gx = cv2.filter2D(img, -1, kern)
	gy = cv2.filter2D(img, -1, kernTrans)

	# Compute gradient fro magnitude and orientation
	# M(x,y) = sqrt(gx^2 + gy^2)
	magnitude = np.sqrt(np.multiply(gx,gx) + np.multiply(gy,gy));

	# O(x,y) = tan^-1(gy/gx)
	orientation = np.abs(np.arctan2(gy, gx))*180/np.pi;
	# print orientation

	v = slitUpBlocks.overlappingBlocksHistogram(orientation, magnitude)
	eps = 1.5
	vNorm = np.sqrt(pow(LA.norm(v),2) + eps)
	v = v/vNorm
	return v
