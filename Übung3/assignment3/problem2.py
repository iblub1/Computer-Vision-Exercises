import numpy as np
import matplotlib.pyplot as plt
import random

"""This is version 3 of the file"""

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
    data = np.load(path, allow_pickle=True)
    pts_pic_1 = data["pts"][0]
    pts_pic_2 = data["pts"][1]
    feats_pic_1 = data["feats"][0]
    feats_pic_2 = data["feats"][1]

    pts = [pts_pic_1, pts_pic_2]
    feats = [feats_pic_1, feats_pic_2]

    return pts, feats

def min_num_pairs():
    """Return the minimum number of point correspondences. Nach den Folien sind das immer konstant 4 für das LGS."""
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
    # Wie in der Beschreibung angegeben, funktioniert das nur wenn pts1 und 2 aligned sind, sonst stimmten die Indizes nicht.
    #
    k = min_num_pairs()
    index_list = random.sample(range(0, len(pts1)), 4)

    pts1_sub = [pts1[i] for i in index_list]
    pts2_sub = [pts2[i] for i in index_list]

    return pts1_sub, pts2_sub


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
    # The Following implementation follows slide 66:
    # TODO: Untested Code/ Pseudocode following:

    x1 = pts1
    x2 = pts2

    # 1.) s, t, s', t' berechnen
    s1 = 1/2 * np.max(np.linalg.norm(x1))
    t2 = np.mean(x1)

    s2 = 1/2 * np.max(np.linalg.norm(x2))
    t2 = np.mean(x2)
    
    # 2.) T und T' aufstellen
    # TODO: was ist t_x und t_y in den folien?
    T1 = np.array([1/s1, 0, -t1_x/s1], [0, 1/s1, -t1_y/s1], [0,0,1])
    T2 = np.array([1/s2, 0, -t2_x/s2], [0, 1/s2, -t2_y/s2], [0,0,1])

    # 3.) x, x' auf u, u' transformieren
    u1 = np.dot(T1, x1)
    u2 = np.dot(T2, x2)

    # 4.) mit SVD(u') H_quer bestimmen
    u, s, vh = np.linalg.svd(a, full_matrices=False)  # Unterbestimmtes Gleichungssytemen, es gibt keine singulärvektoren die Null sind.
    print(vh.shape)

    h_quer = vh[:,-1]  # h = last right singular vector. Müsste 1x9 rauskommen
    H_quer = h_quer.reshape((3,3))

    # 5.) H_quer auf H transformieren
    H = np.linalg.inv(T2) @ H_quer @ T1

    return np.empty(3,  3)


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

    # TODO: Not tested. Not sure if this is correct

    # Transform pts into homogenous coordinates
    z_h = np.ones((len(pts), 1))
    pts_h = np.append(pts, z_h, axis=1)

    # Apply H Matrix
    pts_h = pts_h @ 3

    # Transform back into cartesian coordinates
    pts_x = pts_h[:, 0] / pts_h[:, 2]  # x / z
    pts_y = pts_h[:, 1] / pts_h[:, 2]  # y / z

    pts_result = np.concatenate(pts_x, pts_y, axis=1)
    assert pts_result.shape = pts.shape

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
    # TODO: Code not tested. Work in progress

    inliners_count = 0
    for i, pt1, pt2 in enumerate(zip(pts1, pts2)):
        # Transform point set 1 to align with point set 2
        pt1_transformed = transform_pts(pt1, H)

        # Compute L2-distance
        distance = np.linalg(pt1 - pt2)

        # If distance < threshold increase count by one
        if distance < threshold:
            inliners_count += 1

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

    idxs1, idxs2 = [], []
    ransac_return = np.empty((3,3))

    return ransac_return, idxs1, idxs2