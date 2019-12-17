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
    print(np.linalg.norm(pts, axis=1).shape)
    s = 1/2 * np.max(np.linalg.norm(pts, axis=1))  # calculate norm for all points and choose maximum

    t_x = np.mean(pts[0])  # mean along x-coordinates
    t_y = np.mean(pts[1])  # mean along x-coordinates

    T = np.array([[1/s, 0, -t_x / s], [0, 1/s, -t_y / s], [0, 0, 1]])

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

    # Transform into homogenous coordinates
    z_h = np.ones((len(pts), 1))
    pts_h = np.append(pts, z_h, axis=1)

    # Multiply homogenous points with transformation matrix T
    pts_h = np.dot(pts_h, T)

    
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

    x_1x_2 = pts1[0] * pts2[0]  # Hier wollen wir glaube ich keine matrix multiplikation sondern die x Koordinate des jweiligen Punktes multiplizieren.
    y_1x_2 = pts1[1] * pts2[0]

    x_1y_2 = pts1[0] * pts2[1]
    y_1y_2 = pts1[1] * pts2[1]

    #ones = np.ones(shape=(pts1.shape[0]))  # 2d init
    ones = np.ones(pts1shape[0])  # 1d init

    # matrix construction on slide 69
    A = np.array([
        x_1x_2, 
        y_1x_2, 
        pts2[0], 
        x_1x_2, 
        y_1y_2,
        pts2[1],
        pts1[0],
        pts1[1],
        ones
        ])

    A = A.T # standard consutrcot hängt die 1d arrays zeilenweise an, wir wollen spaltenweise -> transponieren (theoretisch)

    assert A.shape == (pts1.shape[0], 9)

    return A

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
    
    # Use svd to get eigenvalues
    u, s, vh = np.linalg.svd(F, full_matrices=True)

    # Force the smallest Eigenvalue to be 0 
    index = np.argmin(s)
    s[index] = 0
    s = np.diag(s)  # Numpy returns s as a vector. so we need to make a matrix again

    # Build F_final out of svd results
    F_final = u @ s @ vh
    
    
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
    F_final = np.empty((3, 3))
    
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
    return -1.0

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
    return F.copy()

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
    F = np.empty((3, 3))
    res = -1

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
    ys = np.empty((M, N))

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
    T = np.empty((3, 3))
    
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
        return [-1]


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
    e = np.empty((2, ))

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
    K = np.empty((3, 3))

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
    E = np.empty((3, 3))

    return E