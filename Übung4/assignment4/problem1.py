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

    t_x, t_y = np.mean(pts, axis=0) # mean along x/y-Dimension (2,)
    s = 0.5 * np.max(np.abs(pts))   # max-norm (1,)
    # s = 0.5 * np.max(np.linalg.norm(pts, axis=1))
    print('s = ', s, ' | ', s.shape)
    T = np.array([[1/s, 0 , -t_x/s], [0, 1/s, -t_y/s], [0, 0, 1]]) # conditional matrix (3, 3)

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

    pts_h = np.c_[pts, np.ones(len(pts))].dot(T)

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

    # print('pts1 = ', pts1.shape)
    # print('pts2 = ', pts2.shape)
    
    x,  y  = pts1[:, 0], pts1[:, 1]
    x_, y_ = pts2[:, 0], pts2[:, 1] 
    ones   = np.ones_like(x)

    # print('x, y = ', x.shape, y.shape)

    A = np.array([x * x_, y * x_, x_, x * y_, y * y_, y_, x, y, ones]).T
    # print('A = ', A.shape)

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
    s[2] = 0
    F_final = u @ np.diag(s) @ vh   # Build F_final out of svd results

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

    # Slide 23 to see the process

    # We start with SVD
    U_A, D_A, V_A = np.linalg.svd(A, full_matrices=True)

    print('V_A = ', V_A.shape)

    # Construct F_tilde -> rightmost column vector of V_A
    F_tilde = V_A[:, -1].reshape(3, 3)
    

    # F_tilde = np.array([[V_A[0][8], V_A[1][8], V_A[2][8]], 
    #                     [V_A[3][8], V_A[4][8], V_A[5][8]],
    #                     [V_A[6][8], V_A[7][8], V_A[8][8]]]
    #                   )

    # Use SVD again
    U_F, D_F, V_F = np.linalg.svd(F_tilde, full_matrices=True)
    D_F = np.diag(D_F)             

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
    # TODO

    print('Residual')
    print('x1 = ', x1.shape)
    print('x2 = ', x2.shape)
    print('F  = ', F.shape)

    # conputation x1 * F * x2
    sum_g = 0
    for x_1i, x_2i in zip(x1, x2):
        x_1i = x_1i.reshape(-1, 3)
        x_2i = x_2i.reshape(3, -1)
        # print('x_1i.shape = ', x_1i.shape, ' | x_2i.shape = ', x_2i.shape)
        xFx = x_1i @ F @ x_2i

        abs_xFx = abs(xFx)
        sum_g += abs_xFx
    

    # print('N = ', len(x1))
    g = sum_g / len(x1)
    
    # convert to float, since g has shape (1, 1)
    return float(g)

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
    print('T1 = ', T1.shape, ' | T2 = ', T2.shape)
    F_unc = T1.T @ F @ T2
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
    6. Computation of residuals to check the satisfiability of the result 
    7. ??
    8. Profit
    """

    # 1. use "transform" twice to get T_1 and T_2 for pts1 and pts2
    T_1 = transform(x1)
    T_2 = transform(x2)

    # 2. use "transform_pts" twice to get transformed points pts_h1 and pts_h2
    #   returns vector (homogenous)
    u_1h = transform_pts(x1, T_1) 
    u_2h = transform_pts(x2, T_2)

    # 3. use "create_A" with pts_h1 and pts_h2 as input to get A matrix
    #   for creating of the A matrix it is irrelevant if 
    #   the vectors given are 2D or 3D(homogenous), since
    #   only the first two elements are used
    A = create_A(u_1h, u_2h)

    # 4. use "compute_F" with A to get F_final (this method uses enforce_rank_2
    F_ = compute_F(A) # results in the Matrix F-bar

    # 5. use "denorm" with F, T_1 and T_2 to denormalize F 
    F = denorm(F_, T_1, T_2)
    # F = np.empty((3, 3))

    # 6. Computation of residuals to check the satisfiability of the result
    res = compute_residual(F, u_1h, u_2h)

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

    print('----> LINE_Y <----')
    print('xs = ', xs.shape, ' | F = ', F.shape, ' | pts = ', pts.shape)
    
    # l1 = F * x2 | l2 = F * x1
    l = F.dot(pts.T)
    lx, ly, lz = l[0, :], l[1, :], l[2, :]

    print('l.shape = ', l.shape)
    print('lx = ', lx.shape, ' | ly = ', ly.shape, ' | lz = ', lz.shape)

    # ys = np.array([(lz + lx * xi) / (-ly) for xi in xs]).T
    # print('ys = ', ys.shape)

    '''
        line = dot(F,x)
        # epipolar line parameter and values
        t = linspace(0,n,100)
        lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    '''

    ys = np.array([-(lz + lx * xi) / ly for xi in xs]).T
    print('ys = ', ys.shape)

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

    '''
        Possible Approach
        1. Normalize the datapoints to lie between -1 and 1
           https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        2. Scale the data to have the averag distance sqrt(2) from the center 
    '''
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

    Returns    :
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