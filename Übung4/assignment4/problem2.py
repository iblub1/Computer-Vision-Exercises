import numpy as np
import matplotlib.pyplot as plt

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
    
    # print('SSD | patch1 = ', patch1.shape, ' | patch2 = ', patch2.shape)

    # if input has size (m, m), not (m, m, 1), but we can handle either case
    
    cost_ssd = np.sum((patch1 - patch2)**2)
    # if patch1.ndim == patch2.ndim:
    #     if patch1.ndim == 2: # patches are 2D
    #         cost_ssd = np.sum((patch1 - patch2)**2)
    #     if patch1.ndim == 3: # patches are 3D
    #         cost_ssd = np.sum((patch1[:,:,0] - patch2[:,:,0])**2)
        
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
    assert patch1.shape == patch2.shape
    assert patch1.shape[0] == patch1.shape[1]
    m = patch1.shape[0]

    # According to source it also works to normalize the input first and then correlate if values are between [-1, 1] https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    # Die Frage ist, ob diese Art von Normalisierung auch die richtige ist

    # print('[cost_nc]')
    # print('patch1 = ', patch1.shape, '| patch2 = ', patch2.shape)
    patch1 = (patch1 - np.mean(patch1)) / (np.std(patch1) * len(patch1))
    patch2 = (patch2 - np.mean(patch2)) / (np.std(patch2))

    # np.correlate geht nur mit 1D Vektoren. Außerdem gibt dir das auch einen vektor zurück.
    # cost_nc = np.correlate(patch1.flatten(), patch2.flatten(), 'full')

    # Ich glaube du meinst den pearson korr koefficient
    patch1 = patch1.reshape(m, m)
    patch2 = patch2.reshape(m, m)
    cost_nc = np.corrcoef(patch1, patch2)[1,0]

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
    m = patch1.shape[0]

    cost_val = (1 / m**2) * cost_ssd(patch1, patch2)  + alpha * cost_nc(patch1, patch2)
    
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
    padded_img = input_img.copy()

    '''
        Algorithm:
        1. get number of pixels to pad from the formula [ window_size // 2 ]
        2. pad the image
    '''

    # padding is e.g (11 // 2) = 5
    h_pad = w_pad = window_size // 2

    padded_img = np.lib.pad(input_img, ((h_pad, h_pad),
                                        (w_pad, w_pad)),
                                mode=padding_mode
                           )
    print('[pad_image]')
    print('window_size = ', window_size)
    print('input_img = ', input_img.shape)
    print('padded_img = ', padded_img.shape)
    print('\n')


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

    # print('[compute_disparity] -> begin')
    # print('padded_img_l = [', padded_img_l.shape, ']')
    # print('padded_img_r = [', padded_img_r.shape, ']')
    # print('max_disp = [', max_disp, ']')
    # print('window_size = [', window_size, ']')

    ## Code inspired by:
    # https://github.com/davechristian/Simple-SSD-Stereo/blob/master/stereomatch_SSD.py
    ## 

    height, width = padded_img_l.shape

    # [declaration of disparity-image]
    # disparity-image is an 8-Bit grayscale image
    # with the same dimensions as the original image
    disparity = np.zeros((height, width), np.uint8)

    # [considering the padding]
    # start at half_window and end at img_size - half_window
    half_window = int(window_size / 2)
    
    ## [pixelwise computation of disparity of the image]
    # 
    # Algorithm:
    # 1) iterate over height, width [y, x]
    # 2) iterate over disparities   [d]
    # 2.1) cut out a two windows W_L, W_R of the given window-size
    # 2.2) compute the cost function for W_L and W_R
    # 2.3) save the disparity with the lowest cost-value
    # 3) ?
    # 4) Profit
    
     # iterate over rows (height/y)
    for y in range(half_window, height - half_window):
        # print(".", end="", flush=True)
        print('[y] = ', y)
        
        # iterate over columns (width/x)
        for x in range(half_window, width - half_window):

            # initial value for disparity
            best_d = 0    

            # cost initialized as very high value which will be updated if smaller cost is encountered
            cost_prev = np.inf
            
            # iterate over [0, ..., max_disp]
            for d in range(max_disp + 1):

                # avoid out of bounds exceptions, if the given disparity is bigger than the kernel radius    
                if x - d >= half_window:
                    
                    # patch_l stays the same for different d-Values
                    patch_l = padded_img_l[y - half_window : y + half_window + 1, x - half_window : x + half_window + 1]
                    patch_l = patch_l.reshape((patch_l.shape[0], patch_l.shape[1], 1))

                    # patch_r is the same window in the right imaged shifted by (-d)
                    patch_r = padded_img_r[y - half_window : y + half_window + 1, x - half_window - d : x + half_window - d + 1]
                    patch_r = patch_r.reshape((patch_r.shape[0], patch_r.shape[1], 1))

                    # calculate the cost, given the two patches and the alpha coefficient
                    cost = cost_function(patch_l, patch_r, alpha)
                    if cost < cost_prev:
                        cost_prev = cost
                        best_d = d
                else:
                    break
            # set the disparity for the current pixel
            disparity[y, x] = best_d 

    # remove the padding to get the real disparity-image
    disparity = disparity[half_window : height - half_window, half_window : width - half_window]
    print('disparity = ', disparity.shape)

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

    print('disparity_gt = ', disparity_gt.shape)
    print('disparity_res = ', disparity_res.shape)

    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    #
    # Your code goes here
    #
    
    N = disparity_gt.shape[0] * disparity_gt.shape[1]
    aepe = (1 / N) * np.linalg.norm(disparity_gt - disparity_res, ord=1)

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #

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
        # Answer: sharp objects need small windows, otherwise edges geht washed out
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
        # non-textued area is hard for flow-estimation, because all pixels are the same Make window larger to capture more.
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
