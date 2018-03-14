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


def compute_AUC(X_est, X_pred, y_pred, alpha_range, rho, pixels):
    Pr = []
    Re = []
    for alpha in alpha_range:
        #print('alpha = ', alpha)
        X_res = task2(X_est, X_pred, rho, alpha, pixels)
        Pr.append(evaluate(X_res, y_pred, "precision"))
        Re.append(evaluate(X_res, y_pred, "recall"))
    auc_value = metrics.auc(Re, Pr, True)
    print("auc = ", str(auc_value))
    return auc_value



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
               'traffic': {'alpha': 0.0, 'rho': 10.67}}

    n_pixels = 20
    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)
        alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))
        auc = []

        for pixels in range(1, n_pixels):
            print(names[i] + " " + str(pixels))
            auc.append(compute_AUC(X_est, X_pred, y_pred, alpha_range, params[names[i]]['rho'], pixels))

        plt.figure()
        plt.plot(np.arange(1, n_pixels) ** 2, np.array(auc))
        plt.title("AUC vs Pixels Area " + names[i] + " sequence]")
        plt.xlabel("Area Pixels")
        plt.ylabel("AUC")

        plt.savefig(PlotsDirectory + names[i] + '_Pixels_AUC.png', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()



# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


