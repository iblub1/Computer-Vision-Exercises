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
    t_y = np.mean(pts[1])  # mean along y-coordinates

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
    pts_h = np.append(pts, z_h, axis=1) # append column vector (hom.)

    # Multiply homogenous points with transformation matrix T
    pts_h = np.dot(pts_h, T)
    
    # print('pts_h.shape = ', pts_h.shape)

    
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

    # Hier wollen wir glaube ich keine matrix multiplikation sondern 
    # die x Koordinate des jweiligen Punktes multiplizieren.
    x_1x_2 = pts1[0] * pts2[0]  
    y_1x_2 = pts1[1] * pts2[0]

    x_1y_2 = pts1[0] * pts2[1]
    y_1y_2 = pts1[1] * pts2[1]

    #ones = np.ones(shape=(pts1.shape[0]))  # 2d init
    ones = np.ones(pts1.shape[0])  # 1d init

    # matrix construction on slide 69
    A = np.array([
        x_1x_2,     # x * x' 
        y_1x_2,     # y * x' 
        pts2[0],    #   x'
        x_1x_2,     # x * y'
        y_1y_2,     # y * y'
        pts2[1],    #   y'
        pts1[0],    #   x
        pts1[1],    #   y
        ones        #   1
        ])

    # reshape from (9,) -> (1,9) row vector
    A = A.reshape(-1, 9) 

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

    # UNTESTED :)

    # Slide 23 to see the process

    # We start with SVD
    U_A, D_A, V_A = np.linalg.svd(A, full_matrices=True)

    # Construct F_tilde
    F_tilde = np.array([V_A[0][8], V_A[1][8], V_A[2][8]], 
                        [V_A[3][8], V_A[4][8], V_A[5][8]],
                        [V_A[6][8], V_A[7][8], V_A[8][8]]
                      )

    # Use SVD again
    U_F, D_F, V_F = np.linalg.svd(F_tilde, full_matrices=True)  
    D_F = np.diag(D_F)  # Convert D_F from vector to diagonal matrix              

    # Enforce Rank 2 
    F_final = enforce_rank2(D_F)
 
    # F_final = np.empty((3, 3))
    
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

    # This probably needs debugging 
    sum_g = 0
    for x_1i, x_2i in x1, x2:
        print("Shape of x_1i: ", x_1i.shape)
        print("This should be transposed shape of x_1i: ", (x_1i.T).shape)
        xFx = x_1i.T @ F @ x_2i
        
        abs_xFX = abs(xFx)

        sum_g += abs_xFX 

    g = (1 / x1.shape[0]) * sum_g



    # return -1.0
    return g

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

    # calculation as seen in v7, page 71
    F_unc = T1.T @ F @ T2
    
    # return F.copy()
    return F_unc

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

    """Pseudo-Code:
    1. use "transform" twice to get T_1 and T_2 for pts1 and pts2
    2. use "transform_pts" twice to get transformed points pts_h1 and pts_h2
    3. use "create_A" with pts_h1 and pts_h2 as input to get A matrix
    4. use "compute_F" with A to get F_final (this method uses enforce_rank_2
    5. use "denorm" with F, T_1 and T_2 to denormalize F 
    6. Profit ??
    """

    # 1. use "transform" twice to get T_1 and T_2 for pts1 and pts2
    # TODO

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
    # 1/2: 7 points in minimum, however we have non linear opt. Which goes away if we use 8 points (1: correct, 2: false)
    # 3: False, More points improve accuracy, but everything takes way longer because of RANSAC
    # 4: Correct, the essential matrix however does not
    # 5: Correct, Essential matrix captures the relation between two views
    # 6: False (?) Normally only the determinant of an identity matrix should be one (?)
    # 7: Correct, "The prevailing view (from 8 point algorithm) is extremly susceptible to noise and hence virutally useless for most purposses. . This paper challenges that view, by showing that by preceding the algorithm with a very simplenormalization (translation and scaling) of the coordinates of the matched points, results are obtained comparable with the bestiterative algorithms" (Richard I. Hartley)
    def answer(self):
        return [1, 4, 5, 7]


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

    from scipy.linalg import null_space # Are we allowed to use this??? Wir haben auch schonmal Nullspace irgendwo früher berechnet.
    # Epiploes are the left and right Nullspace of the fundamental matrix (slide 19)
    e = null_space(F)

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

    # Source: lecture 2 slide 6 (??)
    # Nicht sicher ob wir h und w nicht doch aus den Bildern bestimmen sollen statt die defaults zu nehmen

    #ax = f / w  # focal length / image width
    #ay = f / h  # focal length / image height
    #center_x = w / 2  # principal point in pixel (x-coordinate)
    #center_y = h / 2  # principal point in pixel (y-coordinate)
    #s = 0 # Camera skew

    ax = f
    ay = f
    center_x = w
    center_y = h 
    s = 0

    K = np.array([
                [ax, s, center_x],
                [0, ay, center_y], 
                [0, 0, 1]
                ])

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

    # Source: Slide 17: K_1^-T @ E @ K_2^-1 = F
    # I assume that K_1 = K_2 = K (?)
    # Meaning we have: K^-T @ E @ K-1 = F
    # E = K^T @ F @ K
    K = intrinsics_K()
    
    E = K.T @ F @ K

    assert E.shape == (3,3)
    return E