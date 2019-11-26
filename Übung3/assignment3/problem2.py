import numpy as np
import matplotlib.pyplot as plt


def load_pts_features(path):
    """ Load interest points and SIFT features.

    Args:
        path: path to the file pts_feats.npz
    
    Returns:
        pts: coordinate points for two images;
             an array (2,) of numpy arrays (N1, 2), (N2, 2)
        feats: SIFT descriptors for two images;
               an array (2,) of numpy arrays (N1, 128), (N2, 128)
    """

    #
    # Your code here
    #

    pts = [np.empty((123, 2)), np.empty((123, 2))]
    feats = [np.empty((123, 128)), np.empty((123, 128))]

    return pts, feats

def min_num_pairs():
    return np.random.randint(1, 32)

def pickup_samples(pts1, pts2):
    """ Randomly select k corresponding point pairs.
    Note that here we assume that pts1 and pts2 have 
    been already aligned: pts1[k] corresponds to pts2[k].

    This function makes use of min_num_pairs()

    Args:
        pts1 and pts2: point coordinates from Image 1 and Image 2
    
    Returns:
        pts1_sub and pts2_sub: N_min randomly selected points 
                               from pts1 and pts2
    """

    #
    # Your code here
    #

    return None, None


def compute_homography(pts1, pts2):
    """ Construct homography matrix and solve it by SVD

    Args:
        pts1: the coordinates of interest points in img1, array (N, 2)
        pts2: the coordinates of interest points in img2, array (M, 2)
    
    Returns:
        H: homography matrix as array (3, 3)
    """

    #
    # Your code here
    #

    return np.empty(3, 3)


def transform_pts(pts, H):
    """ Transform pst1 through the homography matrix to compare pts2 to find inliners

    Args:
        pts: interest points in img1, array (N, 2)
        H: homography matrix as array (3, 3)
    
    Returns:
        transformed points, array (N, 2)
    """

    #
    # Your code here
    #

    return np.empty(100, 2)


def count_inliers(H, pts1, pts2, threshold=5):
    """ Count inliers
        Tips: We provide the default threshold value, but you’re free to test other values
    Args:
        H: homography matrix as array (3, 3)
        pts1: interest points in img1, array (N, 2)
        pts2: interest points in img2, array (N, 2)
        threshold: scale down threshold
    
    Returns:
        number of inliers
    """

    return np.empty(1)


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """

    return np.empty(1)


def ransac(pts1, pts2):
    """ RANSAC algorithm

    Args:
        pts1: matched points in img1, array (N, 2)
        pts2: matched points in img2, array (N, 2)
    
    Returns:
        best homography observed during RANSAC, array (3, 3)
    """

    #
    # Your code here
    #

    best_H = np.empty((3, 3))

    return best_H


def find_matches(feats1, feats2, rT=0.8):
    """ Find pairs of corresponding interest points with distance comparsion
        Tips: We provide the default ratio value, but you’re free to test other values

    Args:
        feats1: SIFT descriptors of interest points in img1, array (N, 128)
        feats2: SIFT descriptors of interest points in img1, array (M, 128)
        rT: Ratio of similar distances
    
    Returns:
        idx1: list of indices of matching points in img1
        idx2: list of indices of matching points in img2
    """

    idx1 = []
    idx2 = []

    #
    # Your code here
    #

    return idx1, idx2


def final_homography(im1, im2, pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
       im1: image 1, array (413, 481, 3)
       im2: image 2, array (375, 450, 3)
       pts1: the coordinates of interest points in img1, array (N, 2)
       pts2: the coordinates of interest points in img2, array (M, 2)
       feats1: SIFT descriptors of interest points in img1, array (N, 128)
       feats2: SIFT descriptors of interest points in img1, array (M, 128)
    
    Returns:
        ransac_return: refitted homography matrix from ransac fucation, array (3, 3)
        idxs1: list of matched points in image 1
        idxs2: list of matched points in image 2
    """

    #
    # Your code here
    #

    idxs1, idxs2 = [], []
    ransac_return = np.empty((3,3))

    return ransac_return, idxs1, idxs2