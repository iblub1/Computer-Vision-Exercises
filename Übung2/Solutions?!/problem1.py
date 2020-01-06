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


    return gauss_2D / np.sum(np.abs(gauss_2D))


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

    downsample = x[0::2, 0::2]

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

    g_k = gaussian_kernel(fsize, sigma)
    GP = [None] * nlevels
    GP[0] = img
    for i in range(1, nlevels):
        GP[i] = downsample_x2(convolve2d(GP[i-1], g_k, boundary='symm', mode='same'))

    return GP

def template_distance(v1, v2, mode=None):
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
    distance = None

    #
    # TODO
    #
    if mode == 'ssd':
        distance = np.sum((v1[:, :] - v2[:, :])**2)
    elif mode == 'dot_prod':
        distance = 1 - (np.dot(v1.flatten(), v2.flatten().T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    else:
        print('please choose a mode.')

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

    win_h, win_w = feat.shape
    h, w = img.shape
    padding = np.lib.pad(img, ((win_h // 2, win_h - win_h // 2),
                               (win_w // 2, win_w - win_w // 2)),
                        mode='constant')
    score_list = []
    mode = 'ssd'    # 'dot_prod' or 'ssd'
    for row in range(0, h + 1, step):
        for col in range(0, w + 1, step):
            window = padding[row:row + win_h, col:col + win_w]
            score = template_distance(feat, window, mode=mode)
            score_list.append(score)
    
    score_array = np.array(score_list)
    min_score = np.min(score_array, axis=0)

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

    for feat in feats:
        score_scale = []
        g_im_list = []
        feat_list = []
        # min_score = 1e8

        for im in imgs:
            GP = gaussian_pyramid(im, 3, 5, 1.4)
            for g_im in GP:
                score = sliding_window(g_im, feat)
                score_scale.append(score)
                g_im_list.append(g_im)
                feat_list.append(feat)
        
        min_s = np.min(score_scale)
        min_idx = score_scale.index(min_s)
        assert len(score_scale) == len(g_im_list) == len(feat_list)
        match.append((min_s, g_im_list[min_idx], feat_list[min_idx]))

    return match