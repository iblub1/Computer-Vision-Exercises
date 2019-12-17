import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import *


#
# Problem 1
#

import problem1 as p1

def problem1():

    # Note that we also provide corridor image for your reference
    im1 = Image.open("data/p1_001.png")
    im2 = Image.open("data/p1_002.png")
    pts1, pts2 = load_pts("data/p1_pts.dat")

    show_pts(im1, im2, pts1, pts2)

    F, res = p1.estimate_F(pts1, pts2, p1.transform)
    print("Fundamental matrix: ")
    print(F)
    print("Residual: {:6.5f}".format(res))

    e1 = p1.compute_epipole(F.T)
    e2 = p1.compute_epipole(F)
    print("Epipole e1: ", e1)
    print("Epipole e2: ", e2)

    show_epipolar(im1, im2, F, pts1, pts2, p1.line_y)

    E = p1.compute_E(F)
    print("Essential matrix: ")
    print(E)

#
# Problem 2
#
import problem2 as p2

def problem2():
    """Example code implementing the steps in Problem 2"""

    # Given parameter. No need to change
    max_disp = 15

    alpha = p2.optimal_alpha()
    print("Alpha: {:4.3f}".format(alpha))

    # Window size. You can freely change, but it should be an odd number
    window_size = 11

    # from utils.py
    im_left = rgb2gray(load_image("data/p2_left.png"))
    im_right = rgb2gray(load_image("data/p2_right.png"))
    disparity_gt = disparity_read("data/p2_gt.png")
   
    padded_img_l = p2.pad_image(im_left, window_size, padding_mode='symmetric')
    padded_img_r = p2.pad_image(im_right, window_size, padding_mode='symmetric') 

    disparity_res = p2.compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha)
    aepe = p2.compute_aepe(disparity_gt, disparity_res)
    print("AEPE: {:4.3f}".format(aepe))
        
    interp = 'bilinear'
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_title('disparity_gt')
    axs[0].imshow(disparity_gt, vmin=0, vmax=20)
    axs[1].set_title('disparity_res')
    axs[1].imshow(disparity_res, vmin=0, vmax=20)
    plt.show()

if __name__ == "__main__":
    problem1()
    problem2()