import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#
# Helper functions
#

def load_image(path):
    return plt.imread(path)

def rgb2gray(im):
    return np.mean(im, -1)

def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    disparity = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return disparity

def load_pts(fn):
    """Load interest points"""
    pts1 = []
    pts2 = []
    for line in open(fn).readlines():
        pt1, pt2 = line.strip(" \n").split(" ")
        pts1.append([int(x) for x in pt1.split(",")[::-1]])
        pts2.append([int(x) for x in pt2.split(",")[::-1]])
        
    return np.array(pts1), np.array(pts2)

def show_pts(im1, im2, pts1, pts2):
    fig = plt.figure(figsize=(10, 6), dpi= 80, facecolor='w', edgecolor='k')

    (ax1, ax2) = fig.subplots(1, 2)
    
    ax1.imshow(im1)
    ax2.imshow(im2)
    
    for pt1 in pts1:
        ax1.scatter(pt1[0], pt1[1], marker='o')
    
    for pt2 in pts2:
        ax2.scatter(pt2[0], pt2[1], marker='o')
    
    plt.show()

def xy2hom(x):
    col_1 = np.ones((x.shape[0], 1))
    return np.concatenate([np.array(x), col_1], axis=-1)

def show_epipolar(im1, im2, F, pts1, pts2, line_y):
    """Visualisation of epipolar lines.
    Note that you need to provide line_y function
    defined in the assignment task to use this function.
    """
    
    fig = plt.figure(figsize=(10, 6), dpi= 80, facecolor='w', edgecolor='k')
    (ax1, ax2) = fig.subplots(1, 2)
    
    ax1.imshow(im1)
    ax2.imshow(im2)
    
    plt.ylim(im1.size[1], 0)
    plt.xlim(0, im1.size[0])
    
    # visualising points
    for pt1, pt2 in zip(pts1, pts2):
        ax1.scatter(pt1[0], pt1[1], marker='o')
        ax2.scatter(pt2[0], pt2[1], marker='o')
      
    w, h = im1.size
    xs = np.linspace(0, w - 1)

    p1_h = xy2hom(pts1)
    p2_h = xy2hom(pts2)
    
    # Note how line_y is used
    ys1 = line_y(xs, F, p2_h)
    ys2 = line_y(xs, F.T, p1_h)
    
    for y1, y2 in zip(ys1, ys2):
        ax1.plot(xs, y1, color='orange')
        ax2.plot(xs, y2, color='orange')

    plt.show()