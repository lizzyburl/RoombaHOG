import numpy as np
import cv2
import hogOrientationBinning

# Gets the HOG descriptor vector
# Parameters:
#	img: The 64x128 image that we want to get the HOG descriptor for.
def getImageVector(img):
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
	# We want to convert this to degrees in the range [0,180)
	orientation = ((np.arctan2(gy, gx)*180/np.pi)+180)%180;

	return hogOrientationBinning.overlappingBlocksHistogram(orientation, magnitude)
	
