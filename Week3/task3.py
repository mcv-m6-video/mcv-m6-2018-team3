import cv2
import numpy as np
import os
import sys
from utils import *
from hole_filling import hole_filling, hole_filling2
from task1 import task1
from estimator_adaptative import evaluate
from morphology import Opening
from sklearn import metrics
from estimator_adaptative import week2_masks
from morphology import Dilatation, Closing




data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task3/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

def task3(X_est, X_pred, rho, alpha, apply = True):

    mask = week2_masks(X_est, X_pred, rho=rho, alpha=alpha)

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel_opening1 = np.ones((2, 2), np.uint8)

    if apply:
        mask = Closing(mask, kernel_closing)
    mask = hole_filling2(mask, connectivity=8, visualize=False)
    if apply:
        mask =  Opening(mask, kernel_opening)
    else:
        mask = Opening(mask, kernel_opening1)


    return mask


def compute_AUC(X_est, X_pred, y_pred, alpha_range, rho, pixels):
    Pr = []
    Re = []
    for alpha in alpha_range:
        print(alpha)
        X_res = task2(X_est, X_pred, rho, alpha, pixels)
        Pr.append(evaluate(X_res, y_pred, "precision"))
        Re.append(evaluate(X_res, y_pred, "recall"))

    return metrics.auc(Re, Pr, True)



def main():
    data_path = '../../databases'
    PlotsDirectory = '../plots/Week3/task2/'

    if not os.path.exists(PlotsDirectory):
        os.makedirs(PlotsDirectory)

    names = ['highway', 'fall', 'traffic']
    estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
    prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

    a = [{'min': 0, 'max': 40, 'step': 1}, {'min': 0, 'max': 40, 'step': 1}, {'min': 0, 'max': 40, 'step': 1}]

    params = { 'highway': {'alpha': 7.25, 'rho': 0.6},
               'fall': {'alpha': 3.2, 'rho': 0.004},
               'traffic': {'alpha': 10.67, 'rho': 0}}

    n_pixels = 20
    for i in range(len(names)):
        #i = 0
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

        mask3 = task3(X_est, X_pred, params[names[i]]['rho'], params[names[i]]['alpha'], True)
        maskno3 = task3(X_est, X_pred, params[names[i]]['rho'], params[names[i]]['alpha'], False)
        print(names[i] + ": F1 score new = " + str(evaluate(mask3, y_pred, 'f1')))
        print(names[i] + ": F1 score past = " + str(evaluate(maskno3, y_pred, 'f1')))

        pr = evaluate(mask3, y_pred, "precision")
        re = evaluate(mask3, y_pred, "recall")

        pr_no = evaluate(maskno3, y_pred, "precision")
        re_no = evaluate(maskno3, y_pred, "recall")





if __name__ == "__main__":
    main()



# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


