import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from PIL import Image
import matplotlib.pyplot as plt
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
    ## print("Following prints should all have the same shape: ")
    ## print("shape Im: ", im1.shape)
    ## print("shape Ix: ", Ix.shape)
    ## print("shape Iy: ", Iy.shape)
    ## print("shape It: ", It.shape)
    ## print("\n")

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

    # radius of the patch
    w = patch_size // 2

    # image dimensions
    rows = Ix.shape[0]
    cols = Ix.shape[1]

    # Task 8
    if aggregate == 'gaussian':
        G = (gaussian_kernel(patch_size, sigma=sigma)).flatten()
    else:
        G = (np.ones((patch_size, patch_size))).flatten()

    # inspired by this wikipedia article
    # https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
    # and especially this youtube video (dimensions of terms)
    # https://www.youtube.com/watch?v=dVlRiJ-Xz8I


    for r in range(w, rows - w):
        for c in range(w, cols - w):            
            #####################################################
            #  construct the needed structures for calculation  #
            #####################################################

            # current patch of Ix/Iy/It as falttened matrix -> vector 
            Ix_rc = Ix[r - w : r + w + 1, c - w : c + w + 1].flatten()
            Ix_rc = Ix_rc.reshape((Ix_rc.shape[0], 1))  # [patch_size * patch_size, 1]

            Iy_rc = Iy[r - w : r + w + 1, c - w : c + w + 1].flatten()
            Iy_rc = Iy_rc.reshape((Iy_rc.shape[0], 1))  # [patch_size * patch_size, 1]

            It_rc = It[r - w : r + w + 1, c - w : c + w + 1].flatten()
            It_rc = It_rc.reshape((It_rc.shape[0], 1))  # [patch_size * patch_size, 1]


            # concatenate the Ix and the Iy vector
            nabla_I   = np.c_[Ix_rc, Iy_rc]             # [patch_size^2, 2]
            nabla_I_T = nabla_I.T                       # [2, patch_size^2]

            #################################
            #   calculation of U = [u, v]   #
            #################################

            ## current step [ M * U = b => U = M^-1 * b ]

            # M = nabla_I.T * nabla_I
            M = nabla_I_T.dot(nabla_I)     # [2, 2]

            ## right hand side: b = nabla_I.T * (-It)
            b = nabla_I_T.dot(-It_rc)      # [2, 1]

            # next step in calculation: U = inv(M) * nabla_I.T * (-It)
            M_inv = np.linalg.inv(M)       # [2, 2]
            U = M_inv.dot(b)               # [2, 1]
            

            u[r, c] = U[0]
            v[r, c] = U[1]

    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v


## Additional Code
# construct Grid
def get_grid(y, x, homogenous=False):
    # indices of image (row = y, column = x)
    coords = np.indices((y, x)).reshape(2, -1)
    if homogenous: # grid with homogenous coordinates
        grid = np.vstack((coords, np.ones(coords.shape[1])))
        # np.c_[coords, np.ones(coords.shape[1])]
    else:          # grid with non-homogenous coordinates
        grid = coords
    return grid


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
    ## Hint: You may find function griddata from package scipy.interpolate useful
    ## code inspired by: https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315
    ## https://github.com/rajat95/Optical-Flow-Warping-Tensorflow/blob/master/warp.py
    ## https://sergevideo.blogspot.com/2014/11/writing-simple-optical-flow-in-python.html
    ## https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py

    # get image dimensions [y, x]
    im_height, im_width = im.shape
    
    # number of pixel
    N = im_height * im_width

    iy, ix = np.mgrid[0:im_height, 0:im_width]         # int-meshgrid
    fy, fx = np.mgrid[0:im_height:1.0, 0:im_width:1.0] # float-meshgrid

    # add the optical flow to the indices (float)
    fx = fx + u
    fy = fy + v

    points = np.c_[ix.reshape(N, 1), iy.reshape(N, 1)]
    xi = np.c_[fx.reshape(N, 1), fy.reshape(N, 1)]
    values = im.reshape(N, 1)
    im_interpol = griddata(points, values, xi, method='linear', fill_value=0.0)
    im_warp = im_interpol.reshape(im_height, im_width)

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

    G = gaussian_kernel(fsize, sigma)
    g_img = convolve2d(x, G, mode='same', boundary='symm')
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
        GP[i] = downsample_x2(GP[i - 1], fsize, sigma)

    return GP

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

# iterative Lucas-Kanade
def iter_LK(im1, im2, n_iter):
    '''
        Function: 
            apply Lucas-Kanade motion estimation until convergence
        Input:
            im1: input image from first gaussian pyramid
            im2: input image from second gaussian pyramid
            n_iter: number of applications of LK
        Output:
            u: estimated motion field in x direction
            v: estimated motion field in y direction
    
    '''

    # print('iter_LK')
    # print('im1 = ', im1.shape, ' | im2 = ', im2.shape)

    # initialize the cost as infintity (to be overwritten) 
    d = np.inf

    i = 0
    img1 = im1.copy()
    img2 = im2.copy()
    # do, when error is n_iter not reached [and if error is > 0 ]
    while i < n_iter and d > 0:
        Ix, Iy, It = compute_derivatives(img1, img2)
        u, v       = compute_motion(Ix, Iy, It)       # get motion field of current iteration
        img1       = warp(img1, u, v)                 # warp im1_k to im2_k
        
        # check the current cost
        d = compute_cost(im1, im2)
                
        i += 1
        # print('[{}] cost: d = {}'.format(i, d))

    return u, v


# upsample function
def expand(im_in):
    '''
        Function: expand the motion flow im to the next higher solution
                  in the gaussian pyramid (with bilinear interpolation)
        Input:
            im_in: image of size [h, w]

        Output:
            im_out: rescaled image of size [2h, 2w]

    '''
    
    # conveting to PIL-Images to rescale
    im_exp = Image.fromarray(im_in)

    # upsampling the images to the next finer pyramid level (double the size)
    dim1, dim2 = im_exp.size
    im_exp = im_exp.resize((dim1 * 2, dim2 * 2), Image.BILINEAR)

    return np.array(im_exp) # get back the numpy array



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

    # Code inspired by:
    # https://www.youtube.com/watch?v=FhlbUHhNpD4

    # descending indices of the gaussian pyramid 
    # (from small to big image)
    K = len(pyramid1)
    levels = np.arange(K - 1, -1, -1)  # [K - 1, K - 2, ..., 0]

    #######################
    #  intial estimation  #
    #######################

    ##################################################
    # print('Level [{}]'.format(K - 1))

    # get coarsest images from the gaussian pyramid
    # at level (k - 1)
    im1_k = pyramid1[-1].copy()
    im2_k = pyramid2[-1].copy()

    # iterative LK-Algorithm (refine the motion)
    uk, vk = iter_LK(im1_k, im2_k, n_iter)

    # expand image to new resolution
    uk_exp = expand(uk) * 2
    vk_exp = expand(vk) * 2
    ###################################################


    ##############################
    #  coarse to fine iteritons  #
    ##############################

    # iterate over [ K-2, K-3, ..., 0 ] -> all scales except the smallest
    for k in levels[1:]:  
        # print('Level [{}]'.format(k))

        # get images of current scale from the pyramid
        im1_k = pyramid1[k]
        im2_k = pyramid2[k]

        # warp im1 image from pyramid1
        im1_k_warp = warp(im1_k, uk_exp, vk_exp)

        # apply LK for iterative refinement on current resolution
        uk, vk = iter_LK(im1_k_warp, im2_k, n_iter)

        # add current motion fields (uk, vk) from iter_LK to 
        # previous expanded motion fields (uk_exp, vk_exp)
        uk = uk + uk_exp
        vk = vk + vk_exp

        # expand the motion fields (uk, vk) to the next higher
        # scale in the gaussian pyramid (to use in next iteration)
        if k > 0:   # all other level
            uk_exp = expand(uk) * 2
            vk_exp = expand(vk) * 2
        else: # last level 0 (no expansion needed -> original scale)
            uk_exp = uk
            vk_exp = vk

        # print('[', k, ']: uk_exp = ', uk_exp.shape, ' | vk_exp = ', vk_exp.shape)

    # print('[FINAL]: u = ', uk_exp.shape, ' | v = ', vk_exp.shape)

    # set the final parameter 
    # (LK at original resolution)
    u = uk_exp
    v = vk_exp

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