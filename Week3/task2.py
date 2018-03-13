import cv2
import numpy as np
import os
import sys
from utils import *
from hole_filling import hole_filling, hole_filling2
from task1 import task1
from morphology import Opening


data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task1/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

def task2(X_est, X_pred, rho, alpha, pixels):

    results = task1(X_est, X_pred, rho, alpha, connectivity=8)

    kernel = np.ones((pixels, pixels), np.uint8)
    results = Opening(results, kernel)

    return results

def main():
    print("hola")

if __name__ == "__main__":
    main()



# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


