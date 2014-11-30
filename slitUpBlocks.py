import numpy as np

def singleCellHistogram(cellOrientationMatrix, cellMagnitudeMatrix):
	# The bins are as follows:
	# [10, 30, 50, 70, 90, 110, 130, 150, 170]
	bins = np.zeros(9);

	# Cringe, we are using a for loop here. Maybe we can get rid of it later?
	rows, cols = cellOrientationMatrix.shape
	for row in range(0,rows):
		for col in range(0, cols):
			orientation = cellOrientationMatrix[row,col]
			# We want to interpolate the weights, such that a magnitude of 85 degrees would but 1/4 of
			# its weight into bin 70, and 3/4 of its weight into bin 90.
			if orientation < 10:
				leftBin = 8
				rightBin = 0
				orientation = 180 + orientation
			elif orientation > 170:
				leftBin = 8
				rightBin = 0
			else:
				leftBin = np.floor((orientation - 10)/20)
				rightBin = leftBin + 1
			leftWeight = (30 - (orientation - leftBin*20))/20
			rightWeight = 1 - leftWeight
			bins[leftBin] = bins[leftBin] + leftWeight*cellMagnitudeMatrix[row,col]
			bins[rightBin] = bins[rightBin] + rightWeight*cellMagnitudeMatrix[row,col]
	return bins

# We can only call this with 16x16 blocks-- throw an exception if this is not the case.
def singleBlockHistogram(cellOrientationMatrix, cellMagnitudeMatrix):
	row,col = cellOrientationMatrix.shape
	if (row!=16 or col != 16):
		raise Exception("Invalid input size")
	# Essentially, we want to break this up into 4 8x8 blocks.
	v = np.concatenate((singleCellHistogram(cellOrientationMatrix[0:8,0:8],cellMagnitudeMatrix[0:8,0:8]), \
		singleCellHistogram(cellOrientationMatrix[0:8,8:16],cellMagnitudeMatrix[0:8,8:16]), \
		singleCellHistogram(cellOrientationMatrix[8:16,0:8],cellMagnitudeMatrix[8:16,0:8]), \
		singleCellHistogram(cellOrientationMatrix[8:16,8:16],cellMagnitudeMatrix[8:16,8:16])))
	return v

# We can assume that we have a 128 x 64 image.
def overlappingBlocksHistogram(cellOrientationMatrix, cellMagnitudeMatrix):
	row,col = cellOrientationMatrix.shape
	if (row!=128 or col != 64):
		raise Exception("Invalid input size")
	# v is giant 3780-D Vector
	v = np.array([])
	for row in range (0,15):
		for col in range (0,7):
			v = np.concatenate((v, singleBlockHistogram( \
				cellOrientationMatrix[row*8:row*8+16, col*8:col*8+16], \
				cellMagnitudeMatrix[row*8:row*8+16, col*8:col*8+16])))
	return v