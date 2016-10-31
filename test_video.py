# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 01:05:59 2016

@author: ETHNEAT
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def get_mb(x1, y1, x2, y2):
    """return slope m and intercept b
    """
    m = (y2-y1)/(x2-x1)
    b = y1 - m * x1
    return m, b

#gr_x1, gr_x2, gr_y1 = 400, 700, 300
#gl_x1, gl_x2, gl_y1 = 330, 100, 300

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

#    global gr_x1, gr_x2, gr_y1, gl_x1, gl_x2, gl_y1
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    plt.imshow(edges, cmap='gray')

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    y_max, x_max, _ = imshape # 540, 960, 3
    y_max -= 1
    x_max -= 1
    vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(x_max * 0.51, y_max * 0.59), (x_max * 0.49, y_max * 0.59),
                          (x_max * 0.09, y_max),        (x_max * 0.92, y_max       ) ]], dtype=np.int32)
    mask = cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    #    rho = 11 # distance resolution in pixels of the Hough grid
    #    theta = 2.3*np.pi/180 # angular resolution in radians of the Hough grid
    #    threshold = 2     # minimum number of votes (intersections in Hough grid cell)
    #    min_line_length = 2 #minimum number of pixels making up a line
    #    max_line_gap = 3    # maximum gap in pixels between connectable line segments

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 49 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 42 #minimum number of pixels making up a line
    max_line_gap = 155 # maximum gap in pixels between connectable line segments

    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    #print(file, '\t', len(lines))

    # Iterate over the output "lines" and draw lines on a blank image
    right_line_m = []
    right_line_b = []
    right_line_y = []
    left_line_m = []
    left_line_b = []
    left_line_y = []
    if lines is None:
        lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            #print((x1,y1), (x2,y2))
            m,b = get_mb(x1,y1,x2,y2)
            if m > 0:
                right_line_m.append(m)
                right_line_b.append(b)
                right_line_y += [y1,y2]
            else:
                left_line_m.append(m)
                left_line_b.append(b)
                left_line_y += [y1,y2]
            #print(m,b)
            #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
            #plt.imshow(line_image)
            #plt.show()
    right_m = np.mean(right_line_m)
    right_b = np.mean(right_line_b)
    right_y1 = min(right_line_y + [333])
    left_m = np.mean(left_line_m)
    left_b = np.mean(left_line_b)
    left_y1 = min(left_line_y + [333])
    #print('right:',right_m,right_b,right_y1)
    #print('left:',left_m,left_b,left_y1)
    # right line
    if len(right_line_m) > 0 and right_m != 0:
        right_x1 = int((right_y1 - right_b) / right_m)
        right_x2 = int((y_max - right_b) / right_m)
    else:
        right_x1 = right_x2 = 0
        right_y1 = y_max
    cv2.line(line_image,(right_x1,right_y1),(right_x2,y_max),(0,255,0),5)
    #plt.imshow(line_image)
    #plt.show()
    # left line
    if len(left_line_m) > 0 and left_m != 0:
        left_x1 = int((left_y1 - left_b) / left_m)
        left_x2 = int((y_max - left_b) / left_m)
    else:
        left_x1 = left_x2 = 0
        left_y1 = y_max
    cv2.line(line_image,(left_x1,left_y1),(left_x2,y_max),(0,255,0),5)
    #plt.imshow(line_image)
    #plt.show()


    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.55, line_image, 1, 0)

    return lines_edges


white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
yellow_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'challres.mp4'
clip1 = VideoFileClip("challenge.mp4")
challenge_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)




