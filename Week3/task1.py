import cv2
import numpy as np
import os
import sys
from utils import *
from hole_filling import hole_filling, hole_filling2


def task1(X_est, X_pred, rho, alpha, connectivity=4):

    # from week2 we chose the best results
    results = week2_masks(X_est, X_pred, rho=rho, alpha=alpha)

    results = hole_filling2(results, connectivity=connectivity, visualize=False)

    return results

# INPUT: X: is a sequence of images, path: directory to save images.
def write_images(X, path, head_filename):

    path = os.path.join(path, head_filename)

    for i in range(X.shape[0]):
        filename = path + str(i).zfill(6) + '.png'
        cv2.imwrite(filename, X[i])

    return

# INPUT: X: original images, Y: images tranformed
def show_images(X, Y, delay=0):
    assert (X.shape[0] == Y.shape[0])

    if (X is None) and (Y is None):
        print("There is no images to show.")

    for i in range(X.shape[0]):
        cv2.imshow("Original Image", X[i])
        cv2.imshow("Threshold Image", Y[i])
        cv2.waitKey(delay=delay)


def main():
    show = True
    write = False

    data_path = '../../databases'
    PlotsDirectory = '../plots/Week3/task1/'

    if not os.path.exists(PlotsDirectory):
        os.makedirs(PlotsDirectory)

    names = ['highway', 'fall', 'traffic']
    estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
    prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

    params = { 'highway': {'alpha': 7.25, 'rho': 0.6},
               'fall': {'alpha': 3.2, 'rho': 0.004},
               'traffic': {'alpha': 0.0, 'rho': 10.67}}

    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

        if i == 0:
            result = task1(X_est, X_pred, params['highway']['rho'], params['highway']['alpha'])
        elif i==1:
            result = task1(X_est, X_pred, params['fall']['rho'], params['fall']['alpha'])
        else:
            result = task1(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'])

    if show:
        show_images((X_est, result), 0)

    if write:
        write_images(result, '~/','result')


if __name__ == "__main__":
    main()



# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


