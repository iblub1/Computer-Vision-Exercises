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
        # Ist kommutativ, also Reihenfolge ist egal
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
    img = image.copy()

    image_list = []
    for sigma in sigmas:
        LoG_img = convolve2d(img, LoG_kernel(sigma=sigma), mode="same")
        image_list.append(LoG_img)

    images = np.asarray(image_list)
    assert images.shape == (len(sigmas), *image.shape)

    return images


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

    img = image.copy()

    image_list = []
    for sigma in sigmas:
        DoG_img = convolve2d(img, DoG(sigma=sigma), mode="same")
        image_list.append(DoG_img)

    images = np.asarray(image_list)
    assert images.shape == (len(sigmas), *image.shape)

    return images
    


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
    # TODO: Für die Aufgabe sollen wir auch die analytische Ableitung ausrechnen. Ka wo wir die Anhängen sollen.

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]     # Kopiert von oben. In diesem Fall geht x von -4 bis 4

    # Errechnete Formel. Zusätzliche Erklärung hier: http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    LoG = (x**2 + y**2 - 2*sigma**2) / sigma**4 * np.exp(-(x**2 + y**2) / 2*sigma**2)

    assert LoG.shape == (fsize, fsize)

    return LoG


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
    #print(response.shape[0], response.shape[1], response.shape[2])  # Debugging
    n_pics = response.shape[0]
    n_rows = response.shape[1]
    n_cols = response.shape[2]

    pad = 4 # Cutout = index 4 (Wert an der fünften Stelle), weil wir 9x9 Kernel benutzen und lauft Aufgabenstellung nicht padden wollen.

    # Percentile Threshholds
    th_max = np.percentile(response, 99.9)
    th_min = np.percentile(response, 0.1)

    unique_local_extrema = []
    for i in range(pad, n_rows - 4):
        for j in range(pad, n_cols -4):
            # Create 16x9x9 window
            window = response[:, (i-pad):(i+pad+1), (j-pad):(j+pad+1)]  
            assert window.shape == (16,9,9)
            
            # Min und Max der "Achse" über alle Sigmas
            min_val = np.argmin(window[:, 4, 4])
            max_val = np.argmax(window[:, 4, 4])

            # Überprüfe ob extrama in dem Fenster größer sind als Extrema auf der Achse :
            local_mins = np.argwhere(window <= np.amin(window[:, 4, 4]))
            local_maxs = np.argwhere(window >= np.amax(window[:, 4, 4]))

            # Falls nicht, dann handelt es sich bei den Achsenminima, um ein unique extrema.
            if len(local_mins) == 1 and window[local_mins].all() <= th_min:
                unique_local_extrema.append((min_val, i, j))

            if len(local_maxs) == 1 and window[local_maxs].all() >= th_max:
                unique_local_extrema.append((max_val, i, j))

    print("We have: ", len(unique_local_extrema), "unique local extrema in the 0.1 percentile")


    return unique_local_extrema


def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 9x9 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''
    s_1 = np.sqrt(2) * sigma
    s_2 = sigma / np.sqrt(2)

    gk_1 = gaussian_kernel(fsize=9, sigma=s_1)
    gk_2 = gaussian_kernel(fsize=9, sigma=s_2)

    DoG = gk_1 - gk_2
    assert DoG.shape == (9, 9)

    return DoG


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

