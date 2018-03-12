from estimator_adaptative import EstimatorAdaptative
from mpl_toolkits.mplot3d import Axes3D
from grid_search import GridSearch
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
import numpy as np
import os
import sys

data_path = '../../databases'
PlotsDirectory = '../plots/Week2/task2/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

Pr = list()
Re = list()
names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
a = [{'min':4, 'max':20, 'step':1.5}, {'min':1, 'max':10, 'step':1},{'min':1, 'max':20, 'step':1.5}]
rho = [0.69, 0.1,0.9]

for i in range(len(names)):
    if len(sys.argv) > 1:
        i = names.index(str(sys.argv[1]))

    print('computing ' + names[i] +' ...')

    [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

    assert len(sys.argv) > 1
    i = names.index(str(sys.argv[1]))
    alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

    for idx, alpha in enumerate(alpha_range):
        print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
        estPrecision = EstimatorAdaptative(alpha=alpha, rho=rho[i], metric="precision")
        estRecall = EstimatorAdaptative(alpha=alpha, rho=rho[i], metric="recall")
        estPrecision.fit(X_est)
        estRecall.fit(X_est)
        Pr.append(estPrecision.score(X_pred, y_pred))
        Re.append(estRecall.score(X_pred, y_pred))

    plt.figure()
    plt.plot(np.array(Re), np.array(Pr), 'b', label='Precision-Recall')
    plt.title("Precision vs Recall curve [AUC = " + str(round(metrics.auc(Re, Pr), 4)) + "] [" + names[i] + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()

    if len(sys.argv) > 1:
        break

    #Empty lists
    Pr[:] = []
    Re[:] = []

