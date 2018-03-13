import cv2
import numpy as np
import os
import sys
from utils import *
from hole_filling import hole_filling, hole_filling2
from task1 import task1
from morphology import Opening


data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task4/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

def task4(X_est, X_pred, rho, alpha):

    masks = week2_masks(X_est, X_pred, rho, alpha)
    shadows = MOG2(X_pred)
    masks[shadows == 1] = 0

    return masks

def main():
    data_path = '../../databases'
    PlotsDirectory = '../plots/Week3/task1/'

    if not os.path.exists(PlotsDirectory):
        os.makedirs(PlotsDirectory)

    names = ['highway', 'fall', 'traffic']
    estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
    prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
    #alpha = [{'min': 4, 'max': 20, 'step': 1.5}, {'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 20, 'step': 1.5}]
    #ro = [{'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 10, 'step': 1}]

    params = { 'highway': {'alpha': 7.25, 'rho': 0.6},
               'fall': {'alpha': 3.2, 'rho': 0.004},
               'traffic': {'alpha': 0.0, 'rho': 10.67}}

    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

        if i == 0:
            result = task4(X_est, X_pred, params['highway']['rho'], params['highway']['alpha'])
            #write_images(result)
        elif i==1:
            result = task4(X_est, X_pred, params['fall']['rho'], params['fall']['alpha'])
        else:
            result = task4(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'])

        print(str(i) + ": F1 score with shadow = ") #TODO: launch Ivan evaluation
        print(str(i) + ": F1 score without shadow = ")  # TODO: launch Ivan evaluation

if __name__ == "__main__":
    main()

# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


