import cv2
import numpy as np
import os
import sys
from utils import *
from hole_filling import hole_filling, hole_filling2
from estimator_adaptative import *


def task1(X_est, X_pred, rho, alpha, connectivity=4):

    # from week2 we chose the best results
    r1 = week2_masks(X_est, X_pred, rho=rho, alpha=alpha)

    r1 = np.array(r1, dtype=np.uint8)

    #results = hole_filling2(r1, connectivity=connectivity, visualize=False)
    results = r1

    return results, r1


def main():
    show = True
    write = True

    data_path = '../../databases'
    PlotsDirectory = '../plots/Week3/task1/'

    if not os.path.exists(PlotsDirectory):
        os.makedirs(PlotsDirectory)

    names = ['highway', 'fall', 'traffic']
    estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
    prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

    params = {'highway': {'alpha': 7.25, 'rho': 0.6}, 'fall': {'alpha': 3.2, 'rho': 0.004}, 'traffic': {'alpha': 10.67, 'rho': 0.0}}
    connectivity = 4

    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)


        if i == 0:
            results, week2_best_masks = task1(X_est, X_pred, params['highway']['rho'], params['highway']['alpha'], connectivity=connectivity)
            if write:
                write_images(results, PlotsDirectory, 'highway_result_')
                write_images(week2_best_masks, PlotsDirectory, 'highway_best_mask_')
        elif i == 1:
            results, week2_best_masks = task1(X_est, X_pred, params['fall']['rho'], params['fall']['alpha'], connectivity=connectivity)
            if write:
                write_images(results, PlotsDirectory, 'fall_result_')
                write_images(week2_best_masks, PlotsDirectory, 'fall_best_mask_')
        else:
            results, week2_best_masks = task1(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'], connectivity=connectivity)
            if write:
                write_images(results, PlotsDirectory, 'traffic_result_')
                write_images(week2_best_masks, PlotsDirectory, 'traffic_best_mask_')

if __name__ == "__main__":
    main()



# ================== TESTING ================
# show = True
# write = True
#
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
# params = {'highway': {'alpha': 7.25, 'rho': 0.6}, 'fall': {'alpha': 3.2, 'rho': 0.004}, 'traffic': {'alpha': 0.0, 'rho': 10.67}}
#
# for i in range(len(names)):
#     [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
#     [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)
#
#     if i == 0:
#         result = task1(X_est, X_pred, params['highway']['rho'], params['highway']['alpha'])
#         if write:
#             write_images(result, PlotsDirectory, 'highway_result_')
#     elif i == 1:
#         result = task1(X_est, X_pred, params['fall']['rho'], params['fall']['alpha'])
#         if write:
#             write_images(result, PlotsDirectory, 'fall_result_')
#     else:
#         result = task1(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'])
#     if write:
#         write_images(result, PlotsDirectory, 'traffic_result_')



