import os
import numpy as np
from scipy.signal import convolve2d

# new includes of the solution
from PIL import Image
import math

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
    return G / (2 * math.pi * sigma**2)

def load_image(path):
    ''' 
    The input image is a) loaded, b) converted to greyscale, and
     c) converted to numpy array [-1, 1].

    Args:
        path: the name of the inpit image
    Returns:
        img: numpy array containing image in greyscale
    '''
    x = np.array(Image.open(path).convert('L')).astype(np.float)
    x /= 255.0
    x = 2. * (x - 0.5)
    return x

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
    
    laplace_img = convolve2d(image, lap_kernel)
    out = [convolve2d(laplace_img, gaussian_kernel(sigma=sigma)) for sigma in sigmas]
    return np.stack(out, axis=0)

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
    out = [convolve2d(image, LoG_kernel(sigma=sigma)) for sigma in sigmas]
    return np.stack(out, axis=0)

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

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x : _x + 1, -_y : _y + 1]
    G = (-1 + 0.5 * (x**2 + y**2) / sigma**2) * np.exp(-0.5 * (x**2 + y**2))
    return G / (math.pi * sigma**4)


    # return np.random.random((fsize, fsize))

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

    thresholdp = np.percentile(response, 99.9)
    thresholdn = np.percentile(response, 0.1)
    sc, row, col = response.shape # 'sc' means scale
    detected = []
    for r in range(4, row - 4):
        for c in range(4, col - 4):
            for s in range(0, sc):
                point = response[s, r, c]
                window = response[:, r - 4 : r + 5, c - 4 : c + 5]
                if point == window.max() and point > thresholdp:
                    if (window == point).sum() == 1:
                        detected.append((s, r, c))
                        break
                if point == window.min() and point < thresholdn:
                    if (window == point).sum() == 1:
                        detected.append((s, r, c))
                        break
    return detected


def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 9x9 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''
    sigma1 = sigma / np.sqrt(2)
    sigma2 = sigma * np.sqrt(2)
    G1 = gaussian_kernel(fsize=9, sigma=sigma1, norm=False)
    G2 = gaussian_kernel(fsize=9, sigma=sigma2, norm=False)
    return G2 - G1

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
    return np.array([[0, 1, 0], [1, - 4, 1], [0, 1, 0]])


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

        return (1, 2)

