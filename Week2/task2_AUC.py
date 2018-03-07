from estimator_adaptative import EstimatorAdaptative
from mpl_toolkits.mplot3d import Axes3D
from grid_search import GridSearch
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
import numpy as np
import os

data_path = '../../databases'
PlotsDirectory = '../plots/Week2/task2/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

Pr = list()
Re = list()

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=True)

    alpha_range = np.arange(0,11)
    if seq_name == 'highway':
        rho = 0.69
    elif seq_name == 'fall':
        rho = 0.1
    elif seq_name == 'traffic':
        rho = 0.9

    for idx, alpha in enumerate(alpha_range):
        print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
        estPrecision = EstimatorAdaptative(alpha=alpha, rho=rho, metric="precision")
        estRecall = EstimatorAdaptative(alpha=alpha, rho=rho, metric="recall")
        estPrecision.fit(X_est, y_est)
        estRecall.fit(X_est, y_est)
        Pr.append(estPrecision.score(X_pred, y_pred))
        Re.append(estRecall.score(X_pred, y_pred))

    plt.figure()
    plt.plot(np.array(Re), np.array(Pr), 'b', label='Precision-Recall')
    plt.title("Precision vs Recall curve [AUC = " + str(round(metrics.auc(Re, Pr), 4)) + "] [" + seq_name + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()

    #Empty lists
    Pr[:] = []
    Re[:] = []

