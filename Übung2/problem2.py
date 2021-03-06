import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils import *

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supposrts recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    
    #
    # You code here
    #
    images = []
    img_shape = (0, 0)

    for root, dirs, files in os.walk(path):
        for file in files:
            if ext in file: # check if file is of pgm-type
                img_path = os.path.join(root, file)
                img = plt.imread(img_path)   # Read the image
                img_shape = img.shape
                img = img.flatten() # Transform 2D image into vector M = height x width
                images.append(img)

    # print("We have 760 images. We loaded succesfully:", len(images), " images")
    img_array = np.asarray(images)                

    # plt.imshow(img_array[0].reshape(img_shape))  # For Debugging: The pictures have 96x84 pixels or 8064 total.
    # print("Each image has shape: ", img_shape)
    # print("\n")

    return img_array, img_shape

    # return np.random.random((16, 256)), (16, 16)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 2, 4) # SVD is more efficient and we can use it to get the eigenvalues. DONT KNOW if there are more arguments

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    # Structure of X: every of the N rows has M values
    #     / - x_1 - \
    # X = | - ... - | , x_i in R^M
    #     \ - x_N - /

    # center the data by subtracting the mean
    mean_vector = X.mean(axis=0)    # axis = 0 (rows)
    # print("Shape of X matrix: ", X.shape)
    # print("Shape of mean_vector: ", mean_vector.shape)
    # print("Mean vector should have the same length as the X matrix columns")

    # (M,1)-Dimensional Vector will be broadcasted to size (M, N)
    X = X - mean_vector

    # Now we use SVD to calculate our Eigenvalues (variances) and Eigenvectors (direction of variance)
    # N x M -> 

    # transpose the normalized X matrix
    X = X.T
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    #  X = USV => (M,N) = (M,N) x (N,N) x (N,N)
    # print('U = {}'.format(U.shape))     # (M, N)
    # print('S = {}'.format(S.shape))     # (N, N) 
    
    
    N = X.shape[1]
    # L = (1/N) * S^2
    lmb = np.square(S) / N

    # print("We have this mean Eigenvalues: ", S.size)


    
    # return np.random.random((100, 10)), np.random.random(10)

    # Dimenstions of output (M, N) and (N,)
    return U, lmb
#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    ##
    # Formula: argmin_D sum_i^D lambda_i >= eta * sum_i^M lambda_i
    ##

    # 1. step: calculate the sum of the lambdas (variances) for all
    #          M principal component to weight it by the given factor p
    var_sum_M = p * np.sum(s)

    # 2. step: add together principle components (Basis-Vectors) till the sum 
    #          of the Eigenvalues(Lambdas) is above the sum of weighted eigenvalues
    #          from step 1
    var_sum_D = 0

    # index to iterate over -> the i where the condition is not met is D
    i = 0
    while var_sum_D < var_sum_M:
        var_sum_D += s[i]       # add new variance S[i](Lambda) of PC[i](u)
        i += 1
    D_c = i

    # print('D = ', D_c, " | var_sum_D = ", var_sum_D)
    
    v = u[:, :D_c] # v is the new basis of D principal components
    # print('u: ', u.shape)
    # print('v: ', v.shape)

    return v

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """
    # print('PROJECTION')
    # print('U = ', u.shape)
    # print('face_image = ', face_image.shape)

    projection = np.dot(u.T, face_image)
    # print('projection = ', projection.shape)

    reconstruction = np.dot(u, projection)
    # print('reconstruction = ', reconstruction.shape)

    image_out = reconstruction
    # print('image_out = ', image_out.shape)
    
    # im = image_out.reshape((96, 84))
    # plt.imshow(im)
    # plt.show()

    # return np.random.random((256, ))
    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",                                    # Yes -> Page 48
                2: "The fewer principal components we use, the smaller is the re-projection error",                     
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (1, 4)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """

    # D is the number of features
    # print('Y: ', Y.shape)   # (N, D) -> flattened images
    # print('x: ', x.shape)   # (1, D) -> image to compare to
    # print('u: ', u.shape)   # (M, D) -> basis vector of D vectors
    # # number of images to retrieve that are the most similar to thegiven image x
    # print('top_n: ', top_n) 

    ## inspired by: http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf
    ## nicht

    # print('SEARCH')
    
    x_a = project(x, u)   # Get the coefficients from xa = [a_1, ..., a_D]
    # print('x_a = ', x_a.shape)
    
    Y_a = []

    N = Y.shape[0]
    # iterate over all N D-dimensional vectors
    for i in range(N):
        Y_a.append(project(Y[i, :], u))  # a_i coefficients of image Y[i]
    Y_a = np.asarray(Y_a)
    # print('Y_a = ', Y_a.shape)

    diff = x_a - Y_a
    norm = np.linalg.norm(diff, axis=1)
    # print('norm = ', norm.shape)

    ## Sanity check: PASSED
    # min_n = np.argmin(norm)
    # Y_min = Y_a[min_n]
    # im = Y_a[min_n]
    # im2 = x_a
    # ims = np.stack([im, im2], 0)
    # show_images(ims, (96, 84))
    # plt.show()

    
    # retrieve the top_n minimal indices of the differences
    min_idx = norm.argsort()[:top_n]

    # vector of images nearest to reference image
    Y_top = Y[min_idx, :]
    
    # return np.random.random((top_n, 256))
    return Y_top

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    
    x1_a = project(x1, u)
    x2_a = project(x2, u)

    t = np.linspace(x1, x2, N)
    # print('t = ', t.shape)

    x1_inter = x1_a + t * x2_a
    x2_inter = x2_a + t * x1_a
    
    y0 = project(x1_inter, u)
    y1 = project(x2_inter, u)
    Y = np.array([[y0], [y1]])
    # return np.random.random((3, 256))
    # print('Y = ', Y.shape)
    return Y
