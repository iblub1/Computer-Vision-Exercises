import numpy as np

def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    #
    # Your code goes here
    #
    cost_ssd = np.power((patch1 - patch2), 2).sum()

    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """

    #
    # Your code goes here
    #

    v1 = patch1.reshape(1, -1)
    v2 = patch2.reshape(1, -1)
    n_v1 = v1 - v1.mean()
    n_v2 = v2 - v2.mean()

    norm = np.linalg.norm

    cost_nc = (n_v1 * n_v2).sum() / (norm(n_v1) + 1e-8) / (norm(n_v2) + 1e-8)

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    #
    # Your code goes here
    #

    h, w = patch1.shape
    cost_val = cost_ssd(patch1, patch2) / (h * w) + alpha * cost_nc(patch1, patch2)
    
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    #
    # Your code goes here
    #

    hf_size = int((window_size - 1) / 2.0)
    padded_img = np.pad(input_img, hf_size, mode=padding_mode)

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    #
    # Your code goes here
    #

    height_p, width_p = padded_img_l.shape

    height = height_p - (window_size - 1)
    width  = width_p  - (window_size - 1)

    disparity = np.zeros((height, width), dtype=float)

    cost_mat = np.ones((height, width, max_disp + 1), dtype=float) * 200

    for hh1 in range(0, height):
        for ww1 in range(0, width):
            for dd in range(0, max_disp + 1):
                hh2 = hh1
                ww2 = ww1 - dd

                if ww2 < 0:
                    continue

                patch1 = padded_img_l[hh1:hh1+window_size, ww1:ww1+window_size]
                patch2 = padded_img_r[hh2:hh2+window_size, ww2:ww2+window_size]
                cost_mat[hh1, ww1, dd] = cost_function(patch1, patch2, alpha)

    disparity = np.argmin(cost_mat, axis=2)

    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    #
    # Your code goes here
    #
    
    aepe = np.abs(disparity_gt - disparity_res).mean()

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    
    # alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    alpha = -0.01
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (1, 2, 1, 2)
