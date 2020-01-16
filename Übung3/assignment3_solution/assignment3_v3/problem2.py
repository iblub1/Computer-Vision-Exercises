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
    return 4

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

    x = min(len(pts1), len(pts2))
    return np.random.choice(range(x), min_num_pairs(), replace=False)


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
    p1 = np.c_[pts1, np.ones(len(pts1))]
    p2 = np.c_[pts2, np.ones(len(pts2))]
    
    A = np.zeros((2 * p1.shape[0], 9))

    for i in range(0, 2 * p1.shape[0], 2):

        z  = p2[i // 2]
        z_ = p1[i // 2]

        A[i][:3] = z_
        A[i + 1][3:6] = z_
        A[i][6:] = -z_ * z[0]
        A[i + 1][6:] = -z_ * z[1]
    
    _, _, Vh = np.linalg.svd(A.T.dot(A))
    V = Vh.T
    H = np.reshape(V[:, -1], (3, 3))

    if np.linalg.norm(H) != 1.:
        H /= np.linalg.norm(H)
    
    return H


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

    pts = np.c_[pts, np.ones(len(pts))].dot(H.T)
    return np.array([pts[:, 0] / pts[:, 2], pts[: 1] / pts[:, 2]]).T


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

    pts2_ = transform_pts(pts1, H)
    dist = np.linalg.norm(pts2 - pts2_, axis=1)
    return np.sum(dist < threshold)


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """

    return int(np.ceil(np.log(1 - z) / np.log(1 - np.power(w,d))))
    


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

    best_H = None
    max_inliers = -99999999999
    n_iters = ransac_iters()

    for _ in range(n_iters):

        idx_sel = pickup_samples(pts1, pts2)
        H = compute_homography(pts1[idx_sel], pts2[idx_sel])

        num_inliers = count_inliers(H, pts1, pts2)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H

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

    for i, feat in enumerate(feats1):
        dist = np.linalg.norm(feats2 - feat, axis=1)
        dist_sort = np.argsort(dist)
        dist1, dist2 = dist[dist_sort[0]], dist[dist_sort[1]]

        if dist1 / max(1e-6, dist2) < rT:
            idx1 += [i]
            idx2 += [dist_sort[0]]

    return idx1, idx2


def final_homography(pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
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

    idxs1, idxs2 = find_matches(feats1, feats2)
    ransac_return = ransac(pts1[idxs1], pts2[idxs2])

    return ransac_return, idxs1, idxs2