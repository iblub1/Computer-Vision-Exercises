import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    return plt.imread(path)

def save_image(path, arr):
    plt.imsave(path, arr)

def rgb2gray(im):
    return np.mean(im, -1)

#
# Problem 1
#
from problem1 import *

def problem1():
    """Example code implementing the steps in Problem 1"""

    img = load_image('data/a1p1.png')
    display_image(img)

    save_as_npy('a1p1.npy', img)

    img1 = load_npy('a1p1.npy')
    display_image(img1)

    img2 = mirror_horizontal(img1)
    display_image(img2)

    display_images(img1, img2)

#
# Problem 2
#
from problem2 import *

def problem2():
    """Example code implementing the steps in Problem 2"""

    im = load_image("data/castle.png")

    bayer = rgb2bayer(im)
    display_images(im, bayer)

    # scale and crop each channel
    bayer[:, :, 0] = scale_and_crop_x2(bayer[:, :, 0])
    bayer[:, :, 1] = scale_and_crop_x2(bayer[:, :, 1])
    bayer[:, :, 2] = scale_and_crop_x2(bayer[:, :, 2])




    # interpolate
    im_zoom, _, _ = bayer2rgb(bayer)


    display_images(bayer, im_zoom)

    save_image("data/castle_out.png", im_zoom)


#
# Problem 3
#
from problem3 import *

def problem3():
    """Example code implementing the steps in Problem 3"""

    image_pts, world_pts = load_points("data/points.npz")
    A = create_A(image_pts, world_pts)
    P = homogeneous_Ax(A)
    K, R = solve_KR(P)
    c = solve_c(P)

    print("K = ", K)
    print("R = ", R)
    print("c = ", c)


#
# Problem 2
#
from problem4 import *
import scipy.signal as signal
import math

def problem4():
    """Example code implementing the steps in Problem 4"""

    to_uint8 = lambda x: np.uint8(255*x)
    conv2d = lambda im, k: signal.convolve2d(im, k, boundary='wrap', mode='same')

    im = rgb2gray(load_image("data/coins.png"))

    sx, sy, sigma, z = create_sobel()

    im_out = apply_sobel(im, sx, sy)
    save_image("data/coins_01.png", im_out)

    s_alpha = sobel_alpha(sx, sy, 0.5*math.pi)
    im_out = conv2d(im, s_alpha)
    save_image("data/coins_02.png", np.abs(im_out))

    s_alpha = sobel_alpha(sx, sy, 0.8*math.pi)
    im_out = conv2d(im, s_alpha)
    save_image("data/coins_03.png", np.abs(im_out))


if __name__ == "__main__":
    # problem1()
    # problem2()
    # problem3()
    problem4()