from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
import estimator

data_path = '../../databases'
PlotsDirectory = '../plots/Week2/task1/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=True)

    g_estimator = estimator.Estimator()
    g_estimator.fit(X_est)

    y_pred = build_mask(y_pred)

    max_alpha = 15
    alphas = np.arange(0, max_alpha, 0.1)
    n_alpha = len(alphas)
    TP = np.zeros(n_alpha)
    TN = np.zeros(n_alpha)
    FP = np.zeros(n_alpha)
    FN = np.zeros(n_alpha)
    TF = np.zeros(n_alpha)
    F1 = np.zeros(n_alpha)
    Pr = np.zeros(n_alpha)
    Re = np.zeros(n_alpha)

    for i, alpha in enumerate(alphas):
        print(i+1,"/",n_alpha)
        g_estimator.set_alpha(alpha)
        predictions = g_estimator.predict(X_pred)
        PE = pixel_evaluation(predictions, y_pred)

        TP[i] = PE[0]
        TN[i] = PE[1]
        FP[i] = PE[2]
        FN[i] = PE[3]
        TF[i] = PE[4]

        F1[i] = f1_score(PE)
        Pr[i] = precision(PE)
        Re[i] = recall(PE)

    # plot results
    alpha = np.arange(n_alpha)

    plt.figure(1)
    line1, = plt.plot(alphas, TP, 'b', label='TP')
    line2, = plt.plot(alphas, TN, 'r', label='TN')
    line3, = plt.plot(alphas, FP, 'g', label='FP')
    line4, = plt.plot(alphas, FN, 'c', label='FN')
    line5, = plt.plot(alphas, TF, 'k', label='TF')
    plt.title("Pixel Evaluation [" + seq_name + " sequence]")
    plt.xlabel("alpha")
    plt.ylabel("Number of Pixels")
    plt.legend(handles=[line1, line2, line3, line4, line5], loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.savefig(PlotsDirectory + seq_name + '_pixel_evaluation_one_gaussian.png', bbox_inches='tight')
    plt.close()

    plt.figure(2)
    line6, = plt.plot(alphas, Pr, 'r', label='Precision')
    line7, = plt.plot(alphas, Re, 'g', label='Recall')
    line8, = plt.plot(alphas, F1, 'b', label='F1')
    plt.title("Global evaluation [" + seq_name + " sequence]")
    plt.xlabel("alpha")
    plt.legend(handles=[line6, line7, line8], loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.savefig(PlotsDirectory + seq_name + '_global_evaluation_one_gaussian.png', bbox_inches='tight')
    plt.close()

    plt.figure(3)
    plt.plot(Re, Pr, 'b', label='Precision-Recall')
    plt.title("Precision vs Recall curve [AUC = " + str(round(metrics.auc(Re, Pr), 4)) +"] [" + seq_name + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig(PlotsDirectory + seq_name + '_precision_recall_one_gaussian.png', bbox_inches='tight')
    plt.close()

    best_f1 = np.argmax(F1)
    print('best_alpha: ' + str(alphas[best_f1]))
    print('best_metric: ' + str(F1[best_f1]))