import numpy as np
from scipy.signal import convolve2d as conv2d
from scipy import interpolate


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
    
    dy = np.zeros((3, 3))
    dy[0, 1] =  0.5
    dy[2, 1] = -0.5

    Ix = conv2d(im1, dy.T)
    Iy = conv2d(im1, dy)
    It = im2 - im1

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

    Ixy = Ix * Iy
    Ixx = Ix * Ix
    Iyy = Iy * Iy

    Ixt = Ix * It
    Iyt = Iy * It

    if aggregate == 'gaussian':
        k = gaussian_kernel(patch_size, sigma)
    else:
        kz = patch_size * patch_size
        k = np.ones((patch_size, patch_size)) / kz

    Axy = conv2d(Ixy, k)
    Axx = conv2d(Ixx, k)
    Ayy = conv2d(Iyy, k)

    Bxt = -conv2d(Ixt, k)
    Byt = -conv2d(Iyt, k)

    # solving for the motion
    z = Axx * Ayy - Axy * Axy

    assert((np.abs(z) > 0).all())

    v = (Axx * Byt - Axy * Bxt) / z
    u = (Ayy * Bxt - Axy * Byt) / z

    
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

    # lets prepare data for fitting
    h, w = im.shape

    x = np.arange(w, dtype=np.float)
    y = np.arange(h, dtype=np.float)

    xs, ys = np.meshgrid(x, y)

    xs_fwd = (xs + u).flatten()
    ys_fwd = (ys + v).flatten()

    points = np.stack([xs_fwd, ys_fwd], -1)
    im = im.flatten()

    im_warp = interpolate.griddata(points, im, (xs, ys), method='linear', fill_value=0)

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    #
    # Your code here
    #

    h, w = im1.shape
    d = im1 - im2
    d = (d * d).sum() / (h * w)

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

    g_k = gaussian_kernel(fsize, sigma)
    x = conv2d(x, g_k, boundary='symm', mode='same')

    return x[::2, ::2]

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

    return GP[::-1]

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

    nlevels = len(pyramid1)

    for i in range(n_iter):
        # increments for this iteration
        du = np.zeros(pyramid1[0].shape)
        dv = np.zeros(pyramid1[0].shape)

        for l, p1, p2 in zip(range(nlevels), pyramid1, pyramid2):
            # upscale the previous OF to current resolution
            du = 2 * resize(du, p1.shape)
            dv = 2 * resize(dv, p1.shape)
            
            # resize OF to current resolution
            iu = resize(u, p1.shape) / 2**(nlevels - l - 1)
            iv = resize(v, p1.shape) / 2**(nlevels - l - 1)

            # estimate of OF so far
            curr_u = iu + du
            curr_v = iv + dv

            p1_warped = warp(p1, curr_u, curr_v)
            cost_before = compute_cost(p1_warped, p2)

            Ix, Iy, It = compute_derivatives(p1_warped, p2)
            est_u, est_v = compute_motion(Ix, Iy, It, aggregate='gaussian')

            p1_warped = warp(p1, curr_u + est_u, curr_v + est_v)
            cost_after = compute_cost(p1_warped, p2)

            print('Cost: {:4.3e} -> {:4.3e}'.format(cost_before, cost_after))

            du += est_u
            dv += est_v

            delta = max(np.abs(du).max(), np.abs(dv).max())
        
        u += du
        v += dv

        im1_warped = warp(im1, u, v)
        ssd = compute_cost(im1_warped, im2)

        print('Iteration {:02d}: SSD  = {:4.3e}, delta = {:4.3e}'.format(i, ssd, delta))
    
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

    """

    return (1, 3, 5)