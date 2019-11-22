import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.signal as signal

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    #
    # TODO
    #

    # 'walk' through all folders of path
    for paths in os.walk(path):        
        
        # load the folders who start with facial
        # ("facial_*", [], [img1, img2, ..., imgN])
        if 'facial' in paths[0]:
            
            # path to directory
            prefix = paths[0]

            # list of file names in directory
            data = paths[2]

            for i in range(len(data)):
                # retrieve full path to file
                # img is already numpy-Array
                img = plt.imread(prefix + '/' + data[i])

                if 'features' in prefix:
                    # plt.imshow(img)
                    # plt.show()
                    feats.append(img)

                if 'images' in prefix:
                    # plt.imshow(img, cmap=plt.cm.gray)
                    # plt.show()
                    imgs.append(img)
                
    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    #
    # TODO
    #

    # using the scipy gaussian function with the given parameters
    gauss = signal.gaussian(fsize, sigma)

    # calculate outer product of the 1D gaussian filter to get a
    # 2D filter

    gauss_2D = np.outer(gauss, gauss)
    
    # normalization factor of the gaussian
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    gauss_2D = norm * gauss_2D

    ## DEBUG-CODE
    # print(gauss_2D)
    # plt.imshow(gauss_2D)
    # plt.colorbar()
    # plt.show()
    ##

    return gauss_2D
- 1
def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    #
    # TODO
    #

    # generate a image of half the size of the input by only using every
    # second pixel
    downsample = x[0::2, 0::2]
        
    ## DEBUG-CODE
    # plt.imshow(downsample)
    # plt.show()
    ## 

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []

    #
    # TODO
    #

    # create gaussian kernel to convolve with the downsampled image
    G = gaussian_kernel(fsize, sigma)

    img_tmp = img.copy()

    # first element of the pyramid is the original image
    GP.append(img_tmp)

    # inspired by:
    # https://cabjudo.github.io/machine_perception/pyramids/

    # start with index 1 since index 0 is already occupied by 
    # the original image
    for level in range(1, nlevels):
        # gaussian smoothing
        img_tmp = convolve2d(img_tmp, G, 'valid')

        # downsample the image
        img_tmp = downsample_x2(img_tmp)

        # save image
        GP.append(img_tmp)


    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''


    #
    # TODO
    #
    import time # For timing calculation needed for multiple choice question


    # DOT PRODUCT


    distance = np.dot(v1, v2)  # use angle between two vectors as distance (dot product)

    # Normalized form with (cos(theta) = v1^T * v2 / |v1||v2|)
    # scaling factors
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)

    # apply normalization factor
    distance = distance / (n_v1 * n_v2)


    # SSD

    s_d_list = []

    ## Formula:
    #  E(I,T) = sum_i,j (I(i,j) - T(i, j))^2
    ##
    """
    for i in range(v1.size):
        s_d = np.square(v1[i] - v2[i])
        s_d_list.append(s_d)

    distance = sum(s_d_list)
"""

    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = None

    #
    # TODO
    #

    # TODO Untested code below:

    # Initialize window with size of feature
    window = feat.copy()
    scores = []
    img_r, img_c = img.shape
    win_r, win_c = window.shape

    
    # determine the number of pixels to pad the image
    # get rozizontal and vertical radius to pad
    p_r, p_c = np.ceil(win_r / 2 - 1), np.ceil(win_c / 2 - 1)

    # padding the image to reach all pixels in the image
    # symetric padding means, that the same pixels at the border
    # will be appended in reversed order
    print('p_r = ', p_r, p_c)
    p_img = np.pad(img, ((p_r, p_r), (p_c, p_c)) , 'symmetric')

    # iterate ver image rows and cols, which is now possible 
    # since we used padding

    for r in range(2, img_r, step):
        for c in range(2, img_c, step):
            sub_image = p_img[-p_r:r:p_r, -p_c:c:p_c]

            sub_image = sub_image.flatten()  
            window = window.flatten()

            distance = template_distance(sub_image, window)

            scores.append(distance)

    min_score = min(scores)

    '''
    # Initialize window with size of feature
    window = feat.copy()

    # Calculate how far we can slide by subtracgint the feature size from the image size
    scores = []
    img_rows, img_cols = img.shape
    win_rows, win_cols = window.shape

    #print("Our window has: ", win_rows, "rows. This should be equal or less than the rows of the image: ", img_rows)
    #print("Our window has: ", win_cols, "cols. This should be equal or less than the cols of the image: ", img_cols)

    # Sanity check that our window is actually smaller than our picture
    if win_rows <= img_rows and win_cols <= img_cols:
        rows, cols = img_rows - win_rows, img_cols - win_cols

    else:
        pad_width =  win_rows - img_rows
        pad_height = win_cols - img_cols
        pad = np.max([pad_width, pad_height])

        img = np.pad(img, pad_width=pad, mode="wrap")
        img_rows, img_cols = img.shape

        rows, cols = img_rows - win_rows, img_cols - win_cols


    # Technically we're sliding the picture over the constant window. Same result
    for row in range(rows):
        for col in range(cols):
            sub_image = img[row:row+win_rows:step, col:col+win_cols:step]  # This basically cuts out our window from the picture

            
            sub_image = sub_image.flatten()  # We flatten both matrices into a vector so we canculate distance between
            window = window.flatten()

            distance = template_distance(sub_image, window)

            scores.append(distance)



    #print(scores)
    # Get smallest distance
    min_score = min(scores)

    '''
    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''
        # Idea: ((1, 2) -> the angle can be calculated from the scalar product
        #                  if we get the angle, the length(brightness) isnt needed
        # return (None, None)  # TODO
        return (1, 2)


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)

    #
    # TODO
    #
    nlevels = 3
    fsize = 5
    sigma = 1.4
    total_results = []


    for img in imgs:
        pyramid_imgs = gaussian_pyramid(img, nlevels, fsize, sigma)
        for p_img in pyramid_imgs:
            for feat in feats:
                min_distance = sliding_window(p_img, feat)
                result = [img, p_img, feat, min_distance]
                total_results.append(result)

    """
    img_results = []
    for img in imgs:

        feat_results = []
        for feat in feats:
            pyramid_imgs = gaussian_pyramid(img, nlevels, fsize, sigma)

            p_results = []
            for p_img in pyramid_imgs:
                min_distance = sliding_window(p_img, feat)
                p_results.append(min_distance)

            min_p_result = min(p_results)
            min_p_level = np.argmin(p_results)  # Returns 0,1,2 for level
            p_result = (min_p_level, min_p_result) # This is the lowest distance for a single image feature combi using all pyramid levels.

            feat_results.append(p_result)
        min_feat_result = min(feat_results[1])
        min_feat_arg = np.argmin(feat_results[1])
        feat_result = (min_feat_arg, min_feat_result)

        print("For this image we detect feature number", min_feat_arg)"""


    # TODO Calculate matches out of distances


    #results.sort(key=lambda x: x[3])  # Sort list by distance OPTIONAL

    for img, p_img, feat, min_distance in total_results:
        print(min_distance)

    return match