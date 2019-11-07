import numpy as np
import scipy.signal as signal

# using the convolution operation as seen in main.py
conv2d = lambda im, k: signal.convolve2d(im, k, boundary='wrap', mode='same')

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
	##########################################################################

	# Filter Construction inspired by:
	# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

	# filter radius [1D -> (2 * r + 1) | 2D -> (2 * r + 1, 2 * r + 1)]
	# Here: (2 * 1 + 1) = (3)
	r = 1  		 

	# (3, 1) Array with indices -r, .., 0, ..., r
	x = np.arange(-r , r + 1).reshape(2 * r + 1, -1) # <- ensure shape of (3, 1), not (3,)

	# exponential term exp(-x^2 / 2 * sigma^2), no normalization
	gauss = np.exp( -(np.square(x)) / (2 * np.square(sigma)) )
	
	# DEBUG-INFO:
	# it is possible to recieve the 2D-Kernel by using the outer product
	# this 2D-array can then easily be plotted with imshow

	# ensure the correct size of the 1D-Kernel
	assert gauss.shape == (3, 1)

	'''
	import matplotlib.pyplot as plt
	## PLOT 1D
	
	plt.plot(x, gauss)
	plt.show()

	# PLOT 2D
	kernel_2D = np.outer(gauss, gauss)
	plt.imshow(kernel_2D)
	plt.colorbar()
	plt.show()
	'''

	##########################################################################
	
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
	##########################################################################

	# implementation as seen in the formula on
	# page 33 of l3-filtering_continued

	central_diff = (1/2) * np.array([[1, 0, -1]])

	##########################################################################
	return central_diff

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
	##########################################################################

	# those two values will be result in a approximation of the sobel filters
	# sx, sy
	sigma = 0.85
	z = 4

	# sx = np.zeros((3, 3))
	# sy = np.zeros((3, 3))

	# create gaussian filter with given sigma 
	g = gaussian(sigma)

	# create central difference filter -> (1,3).T <=> (3, 1)
	dx = diff()

	# 1D gaussian filter info
	print('Gaussian Filter:\n', g, g.shape)

	# 1D central differences filter info
	print('Central Differences Filter:\n', dx, dx.shape)
	
	#    (3, 1) x (1, 3) -> (3, 3)-Matrix
	#   / 0.5 \
	#   |  0  | x (g(0), g(1), g(2)) = ...
	#   \-0.5 /
	sx = np.outer(g, dx) * z
	sy = sx.T


	print("sx:\n", sx, sx.shape) 
	print("sy:\n", sy, sy.shape)


	##########################################################################

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
	##########################################################################

	G_x = conv2d(im_norm, sx)
	G_y = conv2d(im_norm, sy)


	### DEBUG-OUTPUT ###
	import matplotlib.pyplot as plt

	plt.imshow(G_x, cmap='Greys')
	plt.show()

	plt.imshow(G_y, cmap='Greys')
	plt.show()
	####################


	im_norm = np.sqrt(np.square(G_x) + np.square(G_y))

	### DEBUG-OUTPUT ###
	plt.imshow(im_norm, cmap='Greys')
	plt.show()
	####################

	##########################################################################
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
