import numpy as np

def transform(pts):
    """Point conditioning: scale and shift points into [-1, 1] range
    as suggested in the lecture.
    
    Args:
        pts: [Nx2] numpy array of pixel point coordinates
    
    Returns:
        T: [3x3] numpy array, such that Tx normalises 
            2D points x (in homogeneous form).
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2

    #
    # Your code goes here
    #
    
    # concatenate points
    s = 0.5 * np.linalg.norm(pts, axis=1).max()
    ty, tx = np.mean(pts, 0)
    T = np.array([[1, 0, -ty], [0, 1, -tx], [0, 0, s]]) / s

    assert T.shape == (3, 3)
    return T


def transform_pts(pts, T):
    """Applies transformation T on 2D points.
    
    Args:
        pts: (Nx2) 2D point coordinates
        T: 3x3 transformation matrix
    
    Returns:
        pts_out: (Nx3) transformed points in homogeneous form
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert T.shape == (3, 3)

    #
    # Your code goes here
    #
    ones_col = np.ones((pts.shape[0], 1))
    pts_h = np.concatenate([pts, ones_col], 1)
    pts_h = pts_h @ T.T
    
    assert pts_h.shape == (pts.shape[0], 3)
    return pts_h

def create_A(pts1, pts2):
    """Create matrix A such that our problem will be Ax = 0,
    where x is a vectorised representation of the 
    fundamental matrix.
        
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points 
    
    Returns:
        A: numpy array
    """
    assert pts1.shape == pts2.shape

    #
    # Your code goes here
    #

    return np.vstack([np.kron(a, b) for a, b in zip(pts1, pts2)])

def enforce_rank2(F):
    """Enforce rank 2 of 3x3 matrix
    
    Args:
        F: 3x3 matrix
    
    Returns:
        F_out: 3x3 matrix with rank 2
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #
    
    uf, sf, vft = np.linalg.svd(F)
    sf[-1] = 0
    F_final = uf @ np.diag(sf) @ vft
    
    assert F_final.shape == (3, 3)
    return F_final

def compute_F(A):
    """Computing the fundamental matrix from F
    by solving homogeneous least-squares problem
    Ax = 0, subject to ||x|| = 1
    
    Args:
        A: matrix A
    
    Returns:
        f: 3x3 matrix subject to rank-2 contraint
    """
    
    #
    # Your code goes here
    #
    
    u, s, vt = np.linalg.svd(A)
    F = vt.T[:, -1].reshape(3, 3)
    F_final = enforce_rank2(F)
    
    assert F_final.shape == (3, 3)
    return F_final

def compute_residual(F, x1, x2):
    """Computes the residual g as defined in the assignment sheet.
    
    Args:
        F: fundamental matrix
        x1,x2: point correspondences
    
    Returns:
        float
    """

    #
    # Your code goes here
    #

    res = np.abs(np.diag(x1 @ F @ x2.T))
    return res.mean()

def denorm(F, T1, T2):
    """Denormalising matrix F using 
    transformations T1 and T2 which we used
    to normalise point coordinates x1 and x2,
    respectively.
    
    Returns:
        3x3 denormalised matrix F
    """

    #
    # Your code goes here
    #
    print('T1 = ', T1.shape, ' | T2 = ', T2.shape, ' | F = ', F.shape)
    return T1.T @ F @ T2

def estimate_F(x1, x2, t_func):
    """Estimating fundamental matrix from pixel point
    coordinates x1 and x2 and normalisation specified 
    by function t_func (t_func returns 3x3 transformation 
    matrix).
    
    Args:
        x1, x2: 2D pixel coordinates of matching pairs
        t_func: normalising function (for example, transform)
    
    Returns:
        F: fundamental matrix
        res: residual g defined in the assignment
    """
    
    assert x1.shape[0] == x2.shape[0]

    #
    # Your code goes here
    #
    
    T1 = t_func(x1)
    T2 = t_func(x2)

    u1 = transform_pts(x1, T1)
    u2 = transform_pts(x2, T2)

    A = create_A(u1, u2)
    uF = compute_F(A)

    print('Check 1: ', np.linalg.norm(A @ uF.reshape(9, 1)))
    print('Check 2: ', np.linalg.norm(uF))

    # compute residual on all points
    res = compute_residual(uF, u1, u2)

    # denormalizing
    F = denorm(uF, T1, T2)

    return F, res


def line_y(xs, F, pts):
    """Compute corresponding y coordinates for 
    each x following the epipolar line for
    every point in pts.
    
    Args:
        xs: N-array of x coordinates
        F: fundamental matrix
        pts: (Mx3) array specifying pixel corrdinates
             in homogeneous form.
    
    Returns:
        MxN array containing y coordinates of epipolar lines.
    """
    N, M = xs.shape[0], pts.shape[0]
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #

    line = (F @ pts.T).T

    a, b, c = line.T
    c = c[:, np.newaxis]
    b = b[:, np.newaxis]
    a = a[:, np.newaxis]

    ys = -(c + a * xs[np.newaxis, :]) / b

    assert ys.shape == (M, N)
    return ys


#
# Bonus tasks
#

import math

def transform_v2(pts):
    """Point conditioning: scale and shift points into [-1, 1] range.
    
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points
    
    Returns:
        T: numpy array, such that Tx conditions 2D (homogeneous) points x.
    
    """
    
    #
    # Your code goes here
    #

    centroid = np.mean(pts, axis=0)
    rms = math.sqrt(np.sum((pts - centroid)**2) / pts.shape[0])
    norm_factor = math.sqrt(2) / rms
    T = np.array([[norm_factor, 0, -norm_factor * centroid[0]], 
                  [0, norm_factor, -norm_factor * centroid[1]],
                  [0, 0, 1]])
    
    return T


"""Multiple-choice question"""
class MultiChoice(object):

    """ Which statements about fundamental matrix F estimation are true?

    1. We need at least 7 point correspondences to estimate matrix F.
    2. We need at least 8 point correspondences to estimate matrix F.
    3. More point correspondences will not improve accuracy of F as long as 
    the minimum number of points correspondences are provided.
    4. Fundamental matrix contains information about intrinsic camera parameters.
    5. One can recover the rotation and translation (up to scale) from the essential matrix 
    corresponding to the transform between the two views.
    6. The determinant of the fundamental matrix is always 1.
    7. Different normalisation schemes (e.g. transform, transform_v2) may have
    a significant effect on estimation of F. For example, epipoles can deviate.
    (Hint for 7): Try using corridor image pair.)

    Please, provide the indices of correct options in your answer.
    """

    def answer(self):
        '''
            1. Contra, because fundamental matric has 8 degrees of freedom. Pro, still works with 7points, but problem is non-linear
            2. Correct, but its actually not advised to use 8 points. Its still noisy because of image discretization. If all point correspondencies have equal degree of noise, everyone can be used.
            3. In practice incorrect, the more points we have the better the estimate, provided that the point correspondencies are not more noisy than the oens we already have.
            4. Correct. The constraint provide that the points are in pixel spcae. So F has information about geometry. In order to align the point correspondencies we need to know the intric porperties of the camera.
            5. Correct. Thats how we decompose the essential matrix. Essential matrix is transofrmation and rotation. Fundamental matrix also has this information.
            6. False. The fundamental matrix has rank 2, but is a 3x3 matrix. 
            7. Correct.
        '''
        return [2, 4, 5, 7]


def compute_epipole(F, eps=1e-8):
    """Compute epipole for matrix F,
    such that Fe = 0.
    
    Args:
        F: fundamental matrix
    
    Returns:
        e: 2D vector of the epipole
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #

    s, u = np.linalg.eig(F)
    smallest = np.argsort(np.abs(s))
    e = u[:, smallest[0]]
    e /= e[-1]

    e = e[:2]

    return e

def intrinsics_K(f=1.05, h=480, w=640):
    """Return 3x3 camera matrix.
    
    Args:
        f: focal length (same for x and y)
        h, w: image height and width
    
    Returns:
        3x3 camera matrix
    """

    #
    # Your code goes here
    #

    fx = fy = f * max(w, h)
    K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])

    return K

def compute_E(F):
    """Compute essential matrix from the provided
    fundamental matrix using the camera matrix (make 
    use of intrinsics_K).

    Args:
        F: 3x3 fundamental matrix

    Returns:
        E: 3x3 essential matrix
    """

    #
    # Your code goes here
    #
    K = intrinsics_K()
    E = K.T @ F @ K

    return E