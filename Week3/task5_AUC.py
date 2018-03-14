import sys
import matplotlib.pyplot as plt
from estimator_adaptative import evaluate
from sklearn import metrics
from utils import *
from task3 import task3

data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task1/'

Pr_A, Re_A, Pr_B, Re_B = list(), list(), list(), list()
FPR_A, TPR_A, FPR_B, TPR_B = list(), list(), list(), list()
names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
a = [{'min':0, 'max':40, 'step':1}, {'min':0, 'max':40, 'step':1},{'min':0, 'max':40, 'step':1}]
rho = [0.599, 0.004,0]

#Modify this option if you want to compute ROC or PR curves
doROC = False


for i in range(len(names)):
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            i = names.index(str(sys.argv[1]))

    print('computing ' + names[i] +' ...')

    [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

    alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

    for idx, alpha in enumerate(alpha_range):
        print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
        X_res_A = task3(X_est, X_pred, rho[i], alpha, True)
        X_res_B = task3(X_est, X_pred, rho[i], alpha, False)

        if doROC:
            FPR_A.append(evaluate(X_res_A, y_pred, "fpr"))
            TPR_A.append(evaluate(X_res_A, y_pred, "tpr"))
            FPR_B.append(evaluate(X_res_B, y_pred, "fpr"))
            TPR_B.append(evaluate(X_res_B, y_pred, "tpr"))
        else:
            Pr_A.append(evaluate(X_res_A, y_pred, "precision"))
            Re_A.append(evaluate(X_res_A, y_pred, "recall"))
            Pr_B.append(evaluate(X_res_B, y_pred, "precision"))
            Re_B.append(evaluate(X_res_B, y_pred, "recall"))


    plt.figure()

    if doROC:
        line4, = plt.plot(np.array(FPR_A), np.array(TPR_A), 'b', label='4-connectivity AUC = ' + str(round(metrics.auc(FPR_A, TPR_A, False), 4)))
        line8, = plt.plot(np.array(FPR_B), np.array(TPR_B), 'r', label='8-connectivity AUC = ' + str(round(metrics.auc(FPR_B, TPR_B, False), 4)))
        plt.title("ROC curve " + names[i] + " sequence]")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        # Empty lists
        FPR_A[:] = []
        TPR_A[:] = []
        FPR_B[:] = []
        TPR_B[:] = []
    else:
        line4, = plt.plot(np.array(Re_A), np.array(Pr_A), 'b', label='4-connectivity AUC = ' + str(round(metrics.auc(Re_A, Pr_A, True), 4)))
        line8, = plt.plot(np.array(Re_B), np.array(Pr_B), 'r', label='8-connectivity AUC = ' + str(round(metrics.auc(Re_B, Pr_B, True), 4)))
        plt.title("Precision vs Recall curve " + names[i] + " sequence]")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # Empty lists
        Pr_A[:] = []
        Re_A[:] = []
        Pr_B[:] = []
        Re_B[:] = []

    plt.legend(handles=[line4,line8], loc='upper center', bbox_to_anchor=(0.5,-0.1))

    if doROC:
        plt.savefig(PlotsDirectory+ names[i]+'_ROCcurve_AUC.png', bbox_inches='tight')
    else:
        plt.savefig(PlotsDirectory + names[i] + '_PRcurve_AUC.png', bbox_inches='tight')
    plt.close()

    if len(sys.argv) > 1:
        break


