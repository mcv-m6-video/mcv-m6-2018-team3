import cv2
import numpy as np
import os
import sys
from utils import *
from scipy.ndimage import binary_fill_holes, generate_binary_structure, binary_closing


# INPUT: 'images' are images in gray scale.
# OUTPU: 'filled_images' all images flood filled.
def hole_filling(images, visualize=False):
    filled_images = []
    #thresholded_images = []


    for j in range(images.shape[0]):
        th, im_th = cv2.threshold(images[j], 58, 255, cv2.THRESH_BINARY_INV)    # cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
                                                                                 # <<<<==== comentar/quitar si las imágenes ya son binarias

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        for i in range(0, w):
            if (mask[0, i] == 0):
                cv2.floodFill(im_floodfill, mask, (i, 0), 255)   # Floodfill from point (i, 0)

            if (mask[h-1, i] == 0):
                cv2.floodFill(im_floodfill, mask, (i, h-1), 255) # Floodfill from point (i, lastrow-1)

        for i in range(0, h):
            if (mask[i, 0] == 0):
                cv2.floodFill(im_floodfill, mask, (0, i), 255)

            if (mask[i, w-1] == 0):
                cv2.floodFill(im_floodfill, mask, (w-1, i), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        filled_images.append(im_out)

        if visualize:
            #cv2.imshow("Original Image", X_pred[j])
            cv2.imshow("New Image", im_th)
            #cv2.imshow("Filled Image", mask*255)
            #cv2.imshow("Floodfilled Image", im_floodfill)
            #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
            cv2.imshow("Foreground", im_out)
            cv2.waitKey(0)

    # change to np.array
    filled_images = np.array(filled_images, dtype=np.uint8)

    return filled_images


def hole_filling2(images, connectivity=4, visualize=False):
    filled_images = []

    if connectivity == 4:
        se = generate_binary_structure(2,1)
    elif connectivity == 8:
        se = generate_binary_structure(2,2)
    else:
        print("Connectivity must be 4 or 8")
        sys.exit(-1)

    for i in range(images.shape[0]):
        #th, im_th = cv2.threshold(images[i], 60, 255, cv2.THRESH_BINARY_INV)

        filled = binary_fill_holes(images[i], se)
        filled = filled.astype(np.uint8)
        filled_images.append(filled)

        # c = binary_closing(im_th, se)
        # c = c.astype(np.uint8)
        # c = c * 255

        if visualize:
            cv2.imshow("Original Image", X_pred[i])
            cv2.imshow("Threshold Image", im_th)
            cv2.imshow("Filled Image", filled*255)
            #cv2.imshow("Closed and Filled Image", c)
            cv2.waitKey(0)  # change by cv2.waitKey(delay=27) for an automatic se<quence

    # change to np.array
    filled_images = np.array(filled_images, dtype=np.uint8)

    return filled_images
# ================== TESTING ================
# data_path = '../../databases'
# PlotsDirectory = '../plots/Week3/task1/'
#
# if not os.path.exists(PlotsDirectory):
#     os.makedirs(PlotsDirectory)
#
# names = ['highway', 'fall', 'traffic']
# estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
# prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
#
# i=0
# [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
# [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


