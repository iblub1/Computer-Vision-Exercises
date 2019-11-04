import numpy as np

def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''

    image_pts = np.empty((100, 3))
    world_pts = np.empty((100, 4))

    array = np.load(path)
    image_pts = array["image"]
    world_pts = array["world"]


    # sanity checks
    assert image_pts.shape[0] == world_pts.shape[0]

    # homogeneous coordinates
    assert image_pts.shape[1] == 3 and world_pts.shape[1] == 4
    return image_pts, world_pts


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """
    N, _ = x.shape
    assert N == X.shape[0]

    A = np.zeros((2*N, 12))

    for i, X_i in enumerate(X):
        A[2*i, 4:8] = -X_i
        A[2*i, 8:] = x[i, 1] * X_i
        A[2*i + 1, 0:4] = X_i
        A[2*i+1, 8:] = -x[i, 0] * X_i
    
    assert A.shape[0] == 2*N and A.shape[1] == 12
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """
    # Compute SVD
    u, s, vh = np.linalg.svd(A)

    # Get smallest singular value (stored in s)
    ss = np.argmin(s)  # Get index of element smalles ev

    # Get correspoinding right singular vector (rows of vh)
    rsv = vh[ss, :]

    # Reshape 12x1 vector into 3x4 vector using np.reshape(3,4)
    h_A = rsv.reshape(3, 4)

    # Return matrix
    return h_A

def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """
    from scipy.linalg import rq
    k, r = rq(P)  # r is upper triangular and k is unitary/orthogonal

    # Check if elements of k are positive and flip if not
    if np.any(k) < 0:
        print("Flipping signs so that elements of k are positive")
        k = np.negative(k)

    return np.empty((3, 3)), np.empty((3, 3))

def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    _, _, vh = np.linalg.svd(P)

    ns = vh[-1,:]  # null space is last row of vh

    c = ns

    # Transform c into non-homogeneous form (3D-Vector)
    #x_homo = np.array([1, 2])  # Test
    x_homo = c

    w_homo = x_homo[-1]
    aug_vector = x_homo / w_homo
    c = aug_vector[:-1].reshape(3,1)  # all but last element
    print("If back-transformation was correct this should print 1: ", aug_vector[-1])

    # Return x
    return c
