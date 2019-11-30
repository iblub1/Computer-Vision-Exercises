import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

"""This is version 3 of the file"""

#
# Hint: you can make use of this function
# to create Gaussian kernels for different sigmas
#
def gaussian_kernel(fsize=7, sigma=1):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        Gaussian kernel
    '''
    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return G / np.sum(G)  # Needed to fix this, since we need normalizing for discrete function!

def load_image(path):
    ''' 
    The input image is a) loaded, b) converted to greyscale, and
     c) converted to numpy array [-1, 1].

    Args:
        path: the name of the inpit image
    Returns:
        img: numpy array containing image in greyscale
    '''
    img = plt.imread(path)  # Read the image

    # Convert to gray scale using this forumlar:  Y' = 0.2989 R + 0.5870 G + 0.1140 B
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # Min Max Normalising von -1 bis 1
    img = 2 * (img - img.min()) / (img.max() - img.min()) - 1

    # Debug
    #plt.imshow(img, cmap="gray")
    #plt.show()

    return img

def smoothed_laplacian(image, sigmas, lap_kernel):
    ''' 
    Then laplacian operator is applied to the image and
     smoothed by gaussian kernels for each sigma in the list of sigmas.


    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    img = image.copy()

    images = []
    for sig in sigmas:
        # Ist Kommunativ, also Reihenfolge ist egal
        my_img = convolve2d(img, lap_kernel, mode="same")  # Apply Laplacian Operator 
        my_img = convolve2d(my_img, gaussian_kernel(sigma=sig), mode="same")  # Apply Gaussian Filter

        images.append(my_img)

    images = np.asarray(images)  # Making List and Converting to array is faster than filling array. 
    #print(images.shape) # Debugging/Testing

    return images

def laplacian_of_gaussian(image, sigmas):
    ''' 
    Then laplacian of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    return np.empty((len(sigmas), *image.shape))

def difference_of_gaussian(image, sigmas):
    ''' 
    Then difference of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    return np.empty((len(sigmas), *image.shape))

def LoG_kernel(fsize=9, sigma=1):
    '''
    Define a LoG kernel.
    Tip: First calculate the second derivative of a gaussian and then discretize it.
    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        LoG kernel
    '''
    return np.random.random((fsize, fsize))

def blob_detector(response):
    '''
    Find unique extrema points (maximum or minimum) in the response using 9x9 spatial neighborhood 
    and across the complete scale dimension.
    Tip: Ignore the boundary windows to avoid the need of padding for simplicity.
    Tip 2: unique here means skipping an extrema point if there is another point in the local window
            with the same value
    Args:
        response: 3 dimensional response from LoG operator in scale space.

    Returns:
        list of 3-tuples (scale_index, row, column) containing the detected points.
    '''
    return []

def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 9x9 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''
    return np.random.random((9, 9))

def laplacian_kernel():
    '''
    Define a 3x3 laplacian kernel.
    Tip1: I_xx + I_yy
    Tip2: There are two possible correct answers.
    Args:
        none

    Returns:
        laplacian kernel
    '''

    """ Finite Differenzen für zweite Ableitung: I_xx = [1 -2 1] und I_yy = [1 -2 1]^T  
    Das dann in 3D sprich Rest mit Nullen auffüllen und beide addieren."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    return kernel


class Method(object):

    # select one or more options
    REASONING = {
        1: 'it is always more computationally efficient',
        2: 'it is always more precise.',
        3: 'it always has fewer singular points',
        4: 'it can be implemented with convolution',
        5: 'All of the above are incorrect.'
    }

    def answer(self):
        '''Provide answer in the return value.
        This function returns a tuple containing indices of the correct answer.
        '''

        return (None, )

