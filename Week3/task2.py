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


def task2(X_est, X_pred, rho, alpha, pixels):

    results = task1(X_est, X_pred, rho, alpha, connectivity=8)

    kernel = np.ones((pixels, pixels), np.uint8)
    results = Opening(results, kernel)

    return results


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
               'traffic': {'alpha': 0.0, 'rho': 10.67}}

    n_pixels = 20
    auc_final = []
    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)
        alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))
        auc = []

        for pixels in range(1, n_pixels):
            print(names[i] + " " + str(pixels))
            auc.append(compute_AUC(X_est, X_pred, y_pred, alpha_range, params[names[i]]['rho'], pixels))

        auc_final.append(np.array(auc))

    plt.figure(1)
    a_line1, = plt.plot(np.arange(1, n_pixels) ** 2, np.array(auc_final[0]), 'r', label='highway')
    a_line2, = plt.plot(np.arange(1, n_pixels) ** 2, np.array(auc_final[1]), 'g', label='fall')
    a_line3, = plt.plot(np.arange(1, n_pixels) ** 2, np.array(auc_final[2]), 'b', label='traffic')
    plt.title("AUC vs Pixels Area")
    plt.xlabel("Area Pixels")
    plt.ylabel("AUC")
    plt.legend(handles=[a_line1, a_line2, a_line3], loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.savefig(PlotsDirectory + 'task2_Pixels_AUC.png', bbox_inches='tight')

if __name__ == "__main__":
    main()



# ================== TESTING ================
#im = hole_filling(images=X_pred, visualize=True)    # Manual sequence: press "Enter" to advance in the sequence
#hole_filling2(images=X_pred, connectivity=8, visualize=True)  # Manual sequence: press "Enter" to advance in the sequence


