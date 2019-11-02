import numpy as np
import scipy.signal as signal

def rgb2bayer(image):
    """Convert image to bayer pattern:
    [B G]
    [G R]

    Args:
        image: Input image as (H,W,3) numpy array

    Returns:
        bayer: numpy array (H,W,3) of the same type as image
        where each color channel retains only the respective 
        values given by the bayer pattern
    """
    assert image.ndim == 3 and image.shape[-1] == 3

    # otherwise, the function is in-place
    bayer = image.copy()

    #
    # You code goes here
    #
    print("PROBLEM 2 BEGINNT HIER: ")
    rows, cols, c = bayer.shape

    # red channel
    bayer[0:rows+1:2, :, 0] = 0
    bayer[:, 0:cols+1:2, 0] = 0

    # blue channel
    bayer[1::2, :, 2] = 0
    bayer[:, 1::2, 2] = 0

    # green channel
    bayer[0:rows+1:2, 0:cols+1:2, 1] = 0
    bayer[1:rows+1:2, 1:cols+1:2, 1] = 0


    print(bayer.ndim)
    print(bayer.shape)

    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    return bayer

def nearest_up_x2(x):
    """Upsamples a 2D-array by a factor of 2 using nearest-neighbor interpolation.

    Args:
        x: 2D numpy array (H, W)

    Returns:
        y: 2D numpy array if size (2*H, 2*W)
    """
    assert x.ndim == 2
    h, w = x.shape

    #
    # You code goes here
    #
    y = np.empty((2*h, 2*w))




    assert y.ndim == 2 and \
            y.shape[0] == 2*x.shape[0] and \
            y.shape[1] == 2*x.shape[1]
    return y

def bayer2rgb(bayer):
    """Interpolates missing values in the bayer pattern.
    Note, green uses nearest neighbour upsampling; red and blue bilinear.

    Args:
        bayer: 2D array (H,W,C) of the bayer pattern
    
    Returns:
        image: 2D array (H,W,C) with missing values interpolated
        green_K: 2D array (3, 3) of the interpolation kernel used for green channel
        redblue_K: 2D array (3, 3) using for interpolating red and blue channels
    """
    assert bayer.ndim == 3 and bayer.shape[-1] == 3

    #
    # You code goes here
    #
    image = bayer.copy()
    rb_k = np.empty((3, 3))
    g_k = np.empty((3, 3))

    assert image.ndim == 3 and image.shape[-1] == 3 and \
                g_k.shape == (3, 3) and rb_k.shape == (3, 3)
    return image, g_k, rb_k

def scale_and_crop_x2(bayer):
    """Upscamples a 2D bayer pattern by factor 2 and takes the central crop.

    Args:
        bayer: 2D array (H, W) containing bayer pattern

    Returns:
        image_zoom: 2D array (H, W) corresponding to x2 zoomed and interpolated 
        one-channel image
    """
    assert bayer.ndim == 2

    #
    # You code goes here
    #
    copy = bayer.copy()
    h, w = copy.shape

    scaled = np.repeat(copy, 2, axis=0)
    scaled = np.repeat(scaled, 2, axis=1)

    start_h, end_h = int(h/2), int(1.5 * h)
    start_w, end_w = int(w/2), int(1.5 * w)

    cropped = scaled[start_h:end_h, start_w:end_w]

    assert cropped.ndim == 2
    return cropped
