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

    print('Loading Points and Features')
    print('Points: ', len(pts), ' | Features: ', len(pts))
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

    # pts1_sub = [pts1[i] for i in index_list]
    # pts2_sub = [pts2[i] for i in index_list]
    pts1_sub = pts1[index_list]
    pts2_sub = pts2[index_list]
    # print('pts1_sub: ', pts1_sub.shape)
    # print('pts2_sub: ', pts2_sub.shape)

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
    # TODO: Untested Code following:

    x1 = pts1
    x2 = pts2
    print('Compute Homography')
    print('x1: ', x1.shape, ' | x2: ', x2.shape)

    # 0.) Transform into homogenous coordinates
    z_h = np.ones((len(x1), 1))
    x1 = np.append(x1, z_h, axis=1)
    
    z_h = np.ones((len(x2), 1))
    x2 = np.append(x2, z_h, axis=1)
    # 0.)-> OK

    # 1.) s, t, s', t' berechnen
    s1 = 1/2 * np.max(np.linalg.norm(x1))
    t1_x, t1_y, _ = np.mean(x1, axis=0)

    s2 = 1/2 * np.max(np.linalg.norm(x2))
    t2_x, t2_y, _ = np.mean(x2, axis=0)
    
    # 2.) T und T' aufstellen
    T1 = np.array([[1/s1, 0, -t1_x/s1], [0, 1/s1, -t1_y/s1], [0,0,1]])
    T2 = np.array([[1/s2, 0, -t2_x/s2], [0, 1/s2, -t2_y/s2], [0,0,1]])

    # 3.) x, x' auf u, u' transformieren
    u1 = np.dot(x1, T1)
    u2 = np.dot(x2, T2)

    # 4.) mit SVD(u') H_quer bestimmen
    u, s, vh = np.linalg.svd(u2, full_matrices=False)  # Unterbestimmtes Gleichungssytemen, es gibt keine singulärvektoren die Null sind.

    #h_quer = vh[:,-1]  # h = last right singular vector. Müsste 1x9 rauskommen (TODO: Überprüfen!)
    #H_quer = h_quer.reshape((3,3))
    H_quer = vh

    # 5.) H_quer auf H transformieren
    H = np.linalg.inv(T2) @ H_quer @ T1

    assert H.shape == (3, 3)

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

    # TODO: Not tested. Not sure if this is correct

    # Transform pts into homogenous coordinates
    z_h = np.ones((len(pts), 1))
    print('z_h.shape = ', z_h.shape)
    print('pts.shape = ', pts.shape)
    print('H.shape = ', H.shape)
    pts_h = np.append(pts, z_h, axis=1)


    # Apply H Matrix
    # print(pts_h.shape)
    # print(H.shape)
    pts_h = pts_h @ H

    # Transform back into cartesian coordinates
    pts_x = pts_h[:, 0] / pts_h[:, 2]  # x / z
    pts_y = pts_h[:, 1] / pts_h[:, 2]  # y / z
    pts_result = np.array([pts_x, pts_y]).T
    print('pts_result = ', pts_result.shape)

    assert pts_result.shape == pts.shape

    # return np.empty(100, 2)
    return pts_result


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

    '''
    inliners_count = 0
    # for i, pt1, pt2 in enumerate(zip(pts1, pts2)):

    for pt1, pt2 in zip(pts1, pts2):
        # Transform point set 1 to align with point set 2
        pt1_transformed = transform_pts(pt1, H)

        # Compute L2-distance
        # distance = np.linalg(pt1 - pt2) -> welche Methode?
        # np.linalg.norm -> standardmaessig L2-Norm 
        distance = np.linalg.norm(pt1_transformed - pt2)

        # If distance < threshold increase count by one
        if distance < threshold:
            inliners_count += 1

    # return np.empty(1)
    return inliners_count
    '''

    ## Big Assumption: Points are already aligned so that we can 
    ##                 calculate the distance for all at once

    # transform points from pts1 to be compared to pts2
    pts1_in_2 = transform_pts(pts1, H)

    # Using the L2-Distance Metric to find the inliers wo lie
    # in the threshold
    d2 = np.linalg.norm(pts1_in_2 - pts2, axis=1)

    # number of inlier: where distance is below the thresholf
    n_inliers = np.sum(d2 < threshold)
    print('# of Inliers: ', n_inliers)

    return n_inliers



def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """
    # Third equation on slide 71
    k = np.log(1-z) / np.log(1-w**d) 

    # k has to be ceiled to the next bigger index to be used as number of iteration
    k = int(np.ceil(k))
    print("Minimum iterations needed for RANSAC: ", k)

    return k


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

    # Placeholder for best Homography (to be updated)
    best_H = np.empty((3, 3))
    n_best_inliers = -1   

    # Here the "Cookbook recipe" from l6-matching-single_view-v0" 
    # on slide 73 is used

    # start RANSAC-Iteration
    x, x_ = np.copy(pts1), np.copy(pts2)
    

    # number of ransac-iterations (default parameters used)
    k = ransac_iters()
    
    print('# of Iterations k = ', k)

    # [ N_min = 4 ] -> already set in the "pickup_samples" function
    # 

    for i in range(k):
        print('RANSAC Iteration [{}]'.format(i))
        # 1.) pick 4 correspondences (samples)
        xs, x_s = pickup_samples(x, x_)

        # 2.) build equations, estimate homography
        H = compute_homography(xs, x_s)
        
        # 3./4./5. all happen in count_inliers
        #---------------------------------------
        # 3.) transform all points x (from first image to second image)
        # 4.) measure distance to all points x’ (in non-homogeneous coordinates!)
        # 5.) count inliers (scale down threshold too!)
        # print('H.shape = ', H.shape)
        # print('xs: ', xs.shape, ' | x_s: ', x_s.shape)
        inliners = count_inliers(H, xs, x_s)

        if inliners > n_best_inliers:
            n_best_inliers = inliners
            best_H = H

        # 6.) re-estimate final homography -> next Iteration
    

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

    # Info:
    # d_SIFT is the minimum distance for a point in the first set to the one in the second,
    # d'_SIFT is the second minimum distance found for the first point
    # d* = d_SIFT / d'_SIFT
    # Coresspondence is Valid if (d* < treshold)
    
    # check if N >= M 
    N, M = feats1.shape[0], feats2.shape[0]
    # => fuer unterschiedliche Dimensionen noch tauglich mane falls len(feats1) != len(feats2) 

    if N >= M: # if feats1 has more features
        for i, f1 in enumerate(feats1):
            # calculate L2-Norm
            d = np.linalg.norm(feats2 - f1, axis=1)

            # get the indices of the two nearest Points
            nearest_n = np.argsort(d)[:2]
            d_SIFT  = d[nearest_n[0]]
            d_SIFT_ = d[nearest_n[1]]

            # print('d_SIFT: ', d_SIFT)
            # print('d_SIFT_: ', d_SIFT_)

            # compute quotient d* (+ preventing division by 0)
            d_ = d_SIFT / max(d_SIFT_, 1e-8)

            if d_ < rT:
                # append feature number of feats1
                idx1.append(i)
                # append feature number of nearest feature in feats2 
                # (corresponding feature)
                idx2.append(nearest_n[0])
    else: # if feats2 has more features
        for i, f2 in enumerate(feats2):
            # calculate L2-Norm
            d = np.linalg.norm(feats1 - f2, axis=1)

            # get the indices of the two nearest Points
            nearest_n = np.argsort(d)[:2]
            d_SIFT  = d[nearest_n[0]]
            d_SIFT_ = d[nearest_n[1]]


            # compute quotient d* (+ preventing division by 0)
            d_ = d_SIFT / max(d_SIFT_, 1e-8)

            if d_ < rT:
                idx1.append(nearest_n[0])
                idx2.append(i)

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
    idxs1, idxs2 = find_matches(feats1, feats2)

    # // TODO
    print(idxs1, idxs2, max(idxs1), max(idxs2))
    print('pts1', pts1.shape)

    # aligned Points (as numpy arrays)
    pts1_aligned, pts2_aligned = pts1[idxs1], pts2[idxs2]
    print('alogned points: ', pts1_aligned.shape, pts2_aligned.shape)

    ransac_return = ransac(pts1_aligned, pts2_aligned)


    return ransac_return, idxs1, idxs2