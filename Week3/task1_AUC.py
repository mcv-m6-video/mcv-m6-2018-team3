import sys
import matplotlib.pyplot as plt
from estimator_adaptative import evaluate
from sklearn import metrics
from utils import *
from task1 import task1

data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task1/'

Pr_h4, Re_h4, Pr_h8, Re_h8 = list(), list(), list(), list()
names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
a = [{'min':0, 'max':40, 'step':1}, {'min':0, 'max':40, 'step':1},{'min':0, 'max':40, 'step':1}]
rho = [0.599, 0.004,0]
for i in range(len(names)):
    if len(sys.argv) > 1:
        i = names.index(str(sys.argv[1]))

    print('computing ' + names[i] +' ...')

    [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

    alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

    for idx, alpha in enumerate(alpha_range):
        print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
        X_res_h4 = task1(X_est, X_pred, rho[i], alpha, connectivity=4)
        X_res_h8 = task1(X_est, X_pred, rho[i], alpha, connectivity=8)
        Pr_h4.append(evaluate(X_res_h4, y_pred, "precision"))
        Re_h4.append(evaluate(X_res_h4, y_pred, "precision"))
        Pr_h8.append(evaluate(X_res_h8, y_pred, "precision"))
        Re_h8.append(evaluate(X_res_h8, y_pred, "precision"))


    plt.figure()
    line4, = plt.plot(np.array(Re_h4), np.array(Pr_h4), 'b', label='4-connectivity AUC = '+str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
    line8, = plt.plot(np.array(Re_h8), np.array(Pr_h8), 'r', label='8-connectivity AUC = '+str(round(metrics.auc(Re_h8, Pr_h8, True), 4)))
    plt.title("Precision vs Recall curve " + names[i] + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(handles=[line4,line8], loc='upper center', bbox_to_anchor=(0.5,-0.1))

    plt.show()
    plt.savefig(PlotsDirectory+ 'AUC.png', bbox_inches='tight')