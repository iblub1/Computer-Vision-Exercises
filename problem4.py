import numpy as np
import scipy.signal as signal

def gaussian(sigma):
	"""Computes (3, 1) array corresponding to a Gaussian filter.
	Normalisation is not required.

	Args:
		sigma: standard deviation used in the exponential

	Returns:
		gauss: numpy (3, 1) array of type float

	"""

	gauss = np.empty((3, 1))

	#
	# You code goes here
	#
	return gauss

def diff():
	"""Returns the derivative part corresponding to the central differences.
	The effect of this operator in x direction on function f should be:

			diff(x, y) = f(x + 1, y) - f(x - 1, y) 

	Returns:
		diff: (1, 3) array (float)
	"""

	#
	# You code goes here
	#

	return np.empty((1, 3))

def create_sobel():
	"""Creates Sobel operator from two [3, 1] filters
	implemented in gaussian() and diff()

	Returns:
		sx: Sobel operator in x-direction
		sy: Sobel operator in y-direction
		sigma: Value of the sigma used to call gaussian()
		z: scaler of the operator
	"""

	sigma = -9999
	z = -9999
	
	#
	# You code goes here
	#
	sx = np.zeros((3, 3))
	sy = np.zeros((3, 3))

	# do not change this
	return sx, sy, sigma, z

def apply_sobel(im, sx, sy):
	"""Applies Sobel filters to a greyscale image im and returns
	L2-norm.

	Args:
		im: (H, W) image (greyscale)
		sx, sy: Sobel operators in x- and y-direction

	Returns:
		norm: L2-norm of the filtered result in x- and y-directions
	"""

	im_norm = im.copy()

	#
	# Your code goes here
	#
	return im_norm


def sobel_alpha(kx, ky, alpha):
	"""Creates a steerable filter for give kx and ky filters and angle alpha.
	The effect the created filter should be equivalent to 
		cos(alpha) I*kx + sin(alpha) I*ky, where * stands for convolution.

	Args:
		kx, ky: (3x3) filters
		alpha: steering angle

	Returns:
		ka: resulting kernel
	"""

	#
	# You code goes here
	#

	return np.empty((3, 3))


"""
This is a multiple-choice question
"""

class EdgeDetection(object):

	# choice of the method
	METHODS = {
				1: "hysteresis",
				2: "non-maximum suppression"
	}

	# choice of the explanation
	# by "magnitude" we mean the magnitude of the spatial gradient
	# by "maxima" we mean the maxima of the spatial gradient
	EFFECT = {
				1: "it sharpens the edges by retaining only the local maxima",
				2: "it weakens edges with high magnitude if connected to edges with low magnitude",
				3: "it recovers edges with low magnitude if connected to edges with high magnitude",
				4: "it makes the edges thicker with Gaussian smoothing",
				5: "it aligns the edges with a dominant orientation"
	}

	def answer(self):
		"""Provide answer in the return value.
		This function returns tuples of two items: the first item
		is the method you will use and the second item is the explanation
		of its effect on the image. For example,
				((2, 1), (1, 1))
		means "hysteresis sharpens the edges by retaining only the local maxima",
		and "non-maximum suppression sharpens the edges by retaining only the local maxima"
		
		Any wrong answer will cancel the correct answer.
		"""

		return ((-1, -1), )
