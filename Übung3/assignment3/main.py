import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""This is version 3 of the file"""

#
# Problem 1
#
import problem1 as p1

def problem1():
    # List of sigmas from 0.25 to 4 in 0.25 steps
    sigmas = np.linspace(0.25, 4, 16).tolist()

    image = p1.load_image("data/goat.jpeg")
    lap_kernel = p1.laplacian_kernel()

    print("Starting Problem 1")
    method1_output = p1.blob_detector(p1.smoothed_laplacian(image, sigmas, lap_kernel))
    method2_output = p1.blob_detector(p1.laplacian_of_gaussian(image, sigmas))
    method3_output = p1.blob_detector(p1.difference_of_gaussian(image, sigmas))
    print("Finished Problem 1. All three numbers of extrema should be kinda similar")


#
# Problem 2
#
import problem2 as p2

def problem2():
    """Example code implementing the steps in Problem 2"""
    
    pts_array, feats_array = p2.load_pts_features('data/pts_feats.npz')

    # points and features for image1 and image2
    pts1, pts2 = pts_array
    fts1, fts2 = feats_array

    # Debugging
    # p2.compute_homography(pts1, pts2)
    print('Points 1: ', pts1.shape)
    print('Points 2: ', pts2.shape)
    print('Feats 1: ', fts1.shape, ' | Feats 2: ', fts2.shape)
    

    # Loading images
    img1 = Image.open('data/img1.png')
    img2 = Image.open('data/img2.png')

    im1 = np.array(img1)
    im2 = np.array(img2)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'ro', markersize=1.3)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.plot(pts2[:, 0], pts2[:, 1], 'ro', markersize=1.3)


    # display algined image
    H, ix1, ix2 = p2.final_homography(pts1, pts2, feats_array[0],
                                      feats_array[1])

    pts1 = pts1[ix1]
    pts2 = pts2[ix2]


    plt.figure(2)
    plt.subplot(1, 3, 1).set_title('Image 1')
    plt.imshow(im1)
    plt.plot(pts1[:, 0],
             pts1[:, 1],
             'ro',
             markersize=2.3,
             markerfacecolor='none')
    plt.subplot(1, 3, 2).set_title('Image 2')
    plt.imshow(im2)
    plt.plot(pts2[:, 0],
             pts2[:, 1],
             'ro',
             markersize=2.3,
             markerfacecolor='none')
    plt.subplot(1, 3, 3).set_title('Algined image 1')

    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]
    im3 = img1.transform(size=(im1.shape[1], im1.shape[0]),
                                     method=Image.PERSPECTIVE,
                                     data=H_inv.ravel(),
                                     resample=Image.BICUBIC)
    plt.imshow(im3)
    plt.show()

if __name__ == "__main__":
    # problem1()
    problem2()
