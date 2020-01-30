import numpy as np
from scipy.signal import convolve2d
import time

######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape
    
    Ix = np.empty_like(im1)
    Iy = np.empty_like(im1)
    It = np.empty_like(im1)

    #
    # Your code here
    #
    
    # Taken from: Lecture 3 (filtering continued) - Slide 39
    # print("Calculating convolutions for derivatives. This might take a while.")
    # D_x = 1/6 * np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # D_y = 1/6 * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Vereinfachte Kernel. Haben kein smoothing, nur die Ableitung
    D_x = 1/2 * np.array([1, 0, -1]).reshape((1,3))
    D_y = 1/2 * np.array([1, 0, -1]).reshape((3,1))

    
    Ix = convolve2d(im1, D_x, mode="same", boundary="symm")
    Iy = convolve2d(im1, D_y, mode="same", boundary="symm")
    It = im2 - im1

    # Debugging
    print("Following prints should all have the same shape: ")
    print("shape Im: ", im1.shape)
    print("shape Ix: ", Ix.shape)
    print("shape Iy: ", Iy.shape)
    print("shape It: ", It.shape)
    print("\n")

    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    #
    # Your code here
    #

    #R = patch_size**2
    #Ix_flat = Ix.flatten()
    #Iy_flat = Iy.flatten()
    #It_flat = It.flatten()

    print("Extremly slow implementation of lucas Kanade. Takes about 5min. You have been warned.")
    print(time.time())

    # Loop through each pixel
    for row_i in range(Ix.shape[0]):
        for col_i in range(Ix.shape[1]):
            term_1 = np.zeros((2,2))
            term_2 = np.zeros((2,1))

            # Loop through window
            for win_r_ii in range(patch_size):
                for win_c_ii in range(patch_size):

                    win_r_i = win_r_ii - int(patch_size/2)
                    win_c_i = win_c_ii - int(patch_size/2)
                    
                    # Check if we're trying to access pixels outside the given picture
                    if (win_r_i+row_i < 0 or win_r_i+row_i >= Ix.shape[0]) or (win_c_i+col_i < 0 or win_c_i+col_i >= Ix.shape[1]):
                        # Set everything to zero if we're at the corner of the picture
                        I_x = 0
                        I_y = 0
                        I_t = 0
                    else:                     
                        # Get derivative values for pixel
                        I_x = Ix[row_i + win_r_i,  col_i + win_c_i]
                        I_y = Iy[row_i + win_r_i,  col_i + win_c_i]
                        I_t = It[row_i + win_r_i,  col_i + win_c_i]

                    # Do the calculations (slide 55). This is the part inside the sum
                    nabla_I = np.array([I_x, I_y]).reshape((2,1))
                    nabla_I_T = np.array([I_x, I_y]).reshape((1,2))

                    dot_1 = np.dot(nabla_I, nabla_I_T)
                    dot_2 = I_t * nabla_I

                    assert dot_1.shape == (2, 2)
                    assert dot_2.shape == (2, 1)
                    
                    term_1 = term_1 + dot_1
                    term_2 = term_2 + dot_2
            
            # Do the calculations (slide 55). This is the part outside the sum
            term_1_I = np.linalg.inv(term_1)

            result = np.dot(term_1_I, term_2)
            assert result.shape == (2, 1)

            u_i = result[0,:]
            v_i = result[1,:]

            # Store result
            u[row_i, col_i] = u_i
            v[row_i, col_i] = v_i
                    
    
    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape
    
    im_warp = np.empty_like(im)
    #
    # Your code here
    #

    print('im = ', im.shape)
    print('u = ', u.shape)
    print('v = ', v.shape)

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    #
    # Your code here
    #

    # The minimized cost function should be SSD. Slide 36
    # Code taken from assignment 4
    d = np.sum((im2 - im1)**2)

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """


    #
    # Your code here
    #

    g_kernel = gaussian_kernel(fsize, sigma)
    g_img = convolve2d(x, g_kernel, mode='same', boundary='symm')
    x = g_img[0::2, 0::2]

    return x

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''
    
    #
    # Your code here
    #

    GP = [None] * nlevels
    GP[0] = img
    for i in range(1, nlevels):
        GP[i] = downsample_x2(img, fsize, sigma)

    return GP

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape
    
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    #
    # Your code here
    #
    
    assert u.shape == im1.shape and \
            v.shape == im1.shape
    return u, v


###############################
#   Multiple-choice question  #
###############################
def task9_answer():
    """
    Which statements about optical flow estimation are true?
    Provide the corresponding indices in a tuple.

    1. For rectified image pairs, we can estimate optical flow 
       using disparity computation methods.
    2. Lucas-Kanade method allows to solve for large motions in a principled way
       (i.e. without compromise) by simply using a larger patch size.
    3. Coarse-to-fine Lucas-Kanade is robust (i.e. negligible difference in the 
       cost function) to the patch size, in contrast to the single-scale approach.
    4. Lucas-Kanade method implicitly incorporates smoothness constraints, i.e.
       that the flow vector of neighbouring pixels should be similar.
    5. Lucas-Kanade is not robust to brightness changes.
 
    Answers: 
    1. Yes, since disparity is a window-based approach. Slide 5
    2. No, Lucas-Kanade assummes small motion (because of Taylor-approximation). Slide 34
    3. ??
    4. Yes, Lucas-Kanade assumes spatial smoothness of the flow (dispartiy values change slowly). ??
    5. Yes, Lucas-Kanade assumes constant brightness. So its not robust vs brightness changes. Slide 27
    """

    return (1, 3, 4, 5)