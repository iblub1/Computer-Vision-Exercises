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


    # Currently not in use! Maybe later?!
    '''
    # construct elements of Nabla I to use later
    Ix2 = np.square(Ix)
    Ixy = Ix * Iy
    Iy2 = np.square(Iy)
    Itx = Ix * It
    Ity = Iy * It

    if aggregate == 'gaussian':
        G = gaussian_kernel(fsize=5, sigma=sigma)
        Ix2 = convolve2d(Ix2, G)
        Ixy = convolve2d(Ixy, G)
        Iy2 = convolve2d(Iy2, G)
    '''


    # inspired by this wikipedia article
    # https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
    # and especially this youtube video (dimensions of terms)
    # https://www.youtube.com/watch?v=dVlRiJ-Xz8I

    for r in range(w, rows - w):
        for c in range(w, cols - w):            
            #####################################################
            #  construct the needed structures for calculation  #
            #####################################################

            # current patch of Ix as falttened matrix -> vector 
            # size: [patch_size * patch_size, 1]
            Ix_rc = Ix[r - w : r + w + 1, c - w : c + w + 1].flatten()
            Ix_rc = Ix_rc.reshape((Ix_rc.shape[0], 1))

            # current patch of Iy as falttened matrix -> vector
            # size: [patch_size * patch_size, 1]
            Iy_rc = Iy[r - w : r + w + 1, c - w : c + w + 1].flatten()
            Iy_rc = Iy_rc.reshape((Iy_rc.shape[0], 1))

            # current patch of It as falttened matrix -> vector
            # size: [patch_size * patch_size, 1]
            It_rc = It[r - w : r + w + 1, c - w : c + w + 1].flatten()
            It_rc = It_rc.reshape((It_rc.shape[0], 1))


            # concatenate the Ix and the Iy vector
            # size: [patch_size^2, 2]
            nabla_I   = np.c_[Ix_rc, Iy_rc]

            # size: [2, patch_size^2]
            nabla_I_T = nabla_I.T
            # print('Nabla_I = ', nabla_I.shape, ' | Nabla_I.T = ', nabla_I_T.shape)


            #################################
            #   calculation of U = [u, v]   #
            #################################

            ## current step [ M * U = b => U = M^-1 * b ]

            ## left hand side: nabla_I.T * nabla_T * U

            # M = nabla_I.T * nabla_I
            # size: [2, 2] = [2, patch_size^2] * [patch_size^2, 2]
            M = nabla_I_T.dot(nabla_I)

            ## right hand side: b = nabla_I.T * (-It)
            b = nabla_I_T.dot(-It_rc)
            # print('b = ', b.shape)

            # next step in calculation: U = inv(M) * nabla_I.T * (-It)
            M_inv = np.linalg.inv(M)
            U = M_inv.dot(b)
            

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
    n = im_height * im_width

    
    iy, ix = np.mgrid[0:im_height, 0:im_width]
    fy, fx = np.mgrid[0:im_height:1.0, 0:im_width:1.0] # float-meshgrid

    # add the optical flow to the indices (float)
    fx = fx + u
    fy = fy + v

    # clamping of values
    fx = np.minimum(np.maximum(fx, 0), im_width)
    fy = np.minimum(np.maximum(fy, 0), im_height)

    points = np.c_[ix.reshape(n, 1), iy.reshape(n, 1)]
    xi = np.c_[fx.reshape(n, 1), fy.reshape(n, 1)]
    values = im.reshape(n, 1)
    im_interpol = griddata(points, values, xi, method='linear')
    im_warp = im_interpol.reshape(im_height, im_width)

    im_warp = im
    assert im_warp.shape == im.shape
    return im_warp
    


def _warp(im, u, v):
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
    
    # h(x, y) = (u + x, v + y) -> im_warp
    rows, cols = im.shape
    grid_indices = (get_grid(rows, cols, homogenous=False)).T   # returns (y, x)-Indices as v-stacked Matrix
    y_indices = grid_indices[:, 0]
    x_indices = grid_indices[:, 1]

    # index-matrix for warped coordinates (same size as origonal grid)
    warp_grid = np.zeros_like(grid_indices)
    y_indices_warp = warp_grid[:, 0]
    x_indices_warp = warp_grid[:, 1]

    # N = Number of indices -> grid_indices.shape[0]
    N = grid_indices.shape[0]
    
    print('----> max_y = ', np.max(grid_indices[:, 0])) # y-column
    print('----> max_x = ', np.max(grid_indices[:, 1])) # x-column


    for i in range(0, N):
        x_i, y_i = x_indices[i], y_indices[i]

        tx = u[y_i, x_i]
        ty = v[y_i, x_i]

        # new indices for warp-grid
        x_indices_warp[i] = x_i + tx
        y_indices_warp[i] = y_i + ty

    print('Indices: ')
    print('x_i_w = ', x_indices_warp.shape, ' | y_i_w = ', y_indices_warp.shape)
    
    # stack up the y- and x-coordinates to be suitable input of griddata
    warp_points = np.stack([y_indices_warp, x_indices_warp])
    print('points = ', warp_points.shape)


    # interpolate real values to discrete values
    im_warp = griddata(warp_points, im, (y_indices, x_indices), method='linear')

    print('im_warp = ', im_warp.shape)
    plt.imshow(im_warp)
    plt.show()


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
    
    # invert elemets so that highest level is original image
    # GP = GP[::-1]

    # for i in range(len(GP)):
    #     print('Level[{}] = {}'.format(i, GP[i].shape))

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
            u: estimated motion flow in x direction
            v: estimated motion flow in y direction
    
    '''

    print('iter_LK')
    print('im1 = ', im1.shape, ' | im2 = ', im2.shape)

    # initialize the cost as infintity (to be overwritten) 
    d = np.inf

    i = 0
    # do, when error is n_iter not reached [and if error is > 0 ]
    while i < n_iter and d > 0:
        Ix, Iy, It = compute_derivatives(im1, im2)
        u, v       = compute_motion(Ix, Iy, It)       # get motion field of current iteration
        im1_warp   = warp(im1, u, v)                  # warp im1_k to im2_k
        
        # check the current cost
        d = compute_cost(im1_warp, im2)
                
        i += 1
        print('[{}] cost: d = {}'.format(i, d))

    return u, v


# upsample function
def expand(u, v, new_size):
    '''
        Function: expand the motion flows u, v to the next higher solution
                  in the gaussian pyramid (with bilinear interpolation)
        Input:
            u: motion flow in x-direction of size [h, w]
            v: motion flow in y-direction of size [h, w]

        Output:
            u_k:  motion flow in x-direction of size [2h, 2w]
            v_k:  motion flow in y-direction of size [2h, 2w]

    '''
    assert u.shape == v.shape

    print('expand(Input)')
    print('u = ', u.shape, ' | v = ', v.shape)

    # convert to PIL-Image to use the resize operation
    u_k = Image.fromarray(u)
    v_k = Image.fromarray(v)
    print('Image: u_k = ', u_k.size, ' | v_k = ', v_k.size)

    # do the expand operation on the motion field of the current level
    u_k = u_k.resize(new_size, resample=Image.BILINEAR)
    v_k = v_k.resize(new_size, resample=Image.BILINEAR)

    # convert back from PIL-Image to numpy-array
    u_k = np.array(u_k)
    v_k = np.array(v_k)

    # scale the expanded image by 2
    u_k *= 2
    v_k *= 2

    print('expand u_k = ', u_k.shape, ' | v_k = ', v_k.shape)



    assert u_k.shape == v_k.shape
    return u_k, v_k



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

    ## Algorithm
    # 1.) get the Gaussian pyramids of both images
    # 2.) compute motion flow between images of level K 
    #     (iterative LK-Algorithm)
    # 3.) upsample the motion flow u_K, v_K to the size of 
    #     the next level (K-1) in the gaussian pyramid
    # 4.) upscale the motion flow by the factor 2
    # 5.) apply warping to the im1 of level K - 1 in the
    #     gaussian pyramid (results in an relatively good
    #     approximation of the image in the gaussian pyramid 
    #     of im2)
    # 6.) do the steps 2.) - 4.) again with the next level K - 1
    #     (compute motion field and add it up to the upscaled 
    #      motion field of the previous level, and expand it to the
    #      next level, and so on ...)

    ## 1.) gaussian pyramids of image1/2 -> DONE

    ## 2.) compute motion flow between images of level K
    ##     (iterative LK-Algorithm)
    

    K = len(pyramid1)              # determine levels of pyramid
    levels = np.arange(K - 1, -1, -1)  # [K - 1, K - 2, ..., 0]
    print('levels = ', levels, ' | K = ', K)


    ##
    # intial estimation
    ##
    im1_K = pyramid1[-1].copy()
    im2_K = pyramid2[-1].copy()

    # iterative LK-Algorithm
    u_K, v_K = iter_LK(im1_K, im2_K, n_iter)
    print('Highest Level')
    print('u_K = ', u_K.shape, ' | v_K = ', v_K.shape)

    # expand image to new resolution
    new_size = pyramid1[K - 1].shape
    print('new size = ', new_size)
    # u_K, v_K = expand(u_K, v_K, new_size)
    print('next level')
    print('u_K = ', u_K.shape, ' | v_K = ', v_K.shape)

    
    # for k in range(K - 2, -1):
    #     print('other k = ', k)


    # for k in levels:
        # im1_k = pyramid1.copy()
        # im2_k = pyramid2.copy()

        # if k > 0:
            # print('{} = {} | {} - 1 = {}'.format(k, pyramid1[k].shape, k - 1, pyramid1[k - 1].shape))

            # # apply ierative LK-Algorithm for level k (get motion flow u, v)
            # u_k, v_k = iter_LK(im1_k, im2_k, n_iter)

            # # expand and scale the motion flow u, v to size of pyramid level (k - 1)
            # new_size = pyramid1[k - 1].shape
            # u_k, v_k = expand(u_k, v_k, new_size)

            

        # else:
            # print('{} = {}'.format(k, pyramid1[k].shape))




    '''
    # start with the highest index of the pyramid (lowest resolution)
    for k in levels:
        print('Level k = [{}]'.format(k))

        # 2.) iterative Lukas-Kanade (for moition refinement) at level k
        #################################################################################

        # setting starting cost to infinity
        d = np.inf

        # get the pyramid images of level k
        im1_k = pyramid1[k].copy()
        im2_k = pyramid2[k].copy()

        # warp image 1 into image 2
        
        for i in range(n_iter):
            Ix, Iy, It = compute_derivatives(im1_k, im2_k)
            u_k, v_k = compute_motion(Ix, Iy, It)   # get motion field of current iteration
            im1_k = warp(im1_k, u_k, v_k)           # warp im1_k to im2_k
            d = compute_cost(im1_k, im2_k)
            print('[{}] cost: d = {}'.format(i, d))
            
        
        #################################################################################

        # 3.) Upsample the motion flow u_K, v_K to the size of 
        #     the next level (K-1) in the gaussian pyramid

        # given: motion flow [u_k, v_k]

        # 3.1) Upsampling to next size (if final level G0 [original image] is not reached)
        # 3.2) Then resizing by 2
        if k > 0:
            new_size = pyramid1[k - 1].shape
            print('{}_size = {} | {}_size = {}'.format(k, pyramid1[k].shape, k - 1, new_size))

            # convert to PIL-Image to use the resize operation
            u_k = Image.fromarray(u_k)
            v_k = Image.fromarray(v_k)

            # do the expand operation on the motion field of the current level
            u_k = u_k.resize(new_size, resample=Image.BILINEAR)
            v_k = v_k.resize(new_size, resample=Image.BILINEAR)

            # convert back from PIL-Image to numpy-array
            u_k = np.array(u_k) * 2
            v_k = np.array(v_k) * 2
            

    print('DONE')
    '''


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