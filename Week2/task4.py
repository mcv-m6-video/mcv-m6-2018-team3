from estimator_adaptative import EstimatorAdaptative
from mpl_toolkits.mplot3d import Axes3D
from grid_search import GridSearch
import matplotlib.pyplot as plt
import matplotlib as mpl
import rgb_estimator
from sklearn import metrics
from utils import *
import numpy as np
import os
"""

data_path = '../../databases'
PlotsDirectory = '../plots/Week2/task4/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=False)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=False)

    g_estimator = rgb_estimator.rgbEstimator()
    g_estimator.fit(X_est, y_est)

    y_pred = build_mask(y_pred)

    n_alpha = 15
    TP = np.zeros(n_alpha)
    TN = np.zeros(n_alpha)
    FP = np.zeros(n_alpha)
    FN = np.zeros(n_alpha)
    TF = np.zeros(n_alpha)
    F1 = np.zeros(n_alpha)
    Pr = np.zeros(n_alpha)
    Re = np.zeros(n_alpha)

    for alpha in range(0, n_alpha):
        g_estimator.set_alpha(alpha)
        predictions = g_estimator.predict(X_pred)
        PE = pixel_evaluation(predictions, y_pred)

        TP[alpha] = PE[0]
        TN[alpha] = PE[1]
        FP[alpha] = PE[2]
        FN[alpha] = PE[3]
        TF[alpha] = PE[4]

        F1[alpha] = f1_score(PE)
        Pr[alpha] = precision(PE)
        Re[alpha] = recall(PE)

    # plot results
    alpha = np.arange(n_alpha)

    plt.figure(1)
    line1, = plt.plot(alpha, TP, 'b', label='TP')
    line2, = plt.plot(alpha, TN, 'r', label='TN')
    line3, = plt.plot(alpha, FP, 'g', label='FP')
    line4, = plt.plot(alpha, FN, 'c', label='FN')
    line5, = plt.plot(alpha, TF, 'k', label='TF')
    plt.title("Pixel Evaluation [" + seq_name + " sequence]")
    plt.xlabel("alpha")
    plt.ylabel("Number of Pixels")
    plt.legend(handles=[line1, line2, line3, line4, line5], loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.savefig(PlotsDirectory + seq_name + '_pixel_evaluation_one_gaussian.png', bbox_inches='tight')
    plt.close()

    plt.figure(2)
    line6, = plt.plot(alpha, Pr, 'r', label='Precision')
    line7, = plt.plot(alpha, Re, 'g', label='Recall')
    line8, = plt.plot(alpha, F1, 'b', label='F1')
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
"""

data_path = '../../databases'
PlotsDirectory = '../plots/Week2/task2/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=False)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=False)

    alpha_range = np.arange(0,8,0.5)
    rho_range = np.arange(2,9)/10

    parameters = {'alpha': alpha_range, 'rho': rho_range}
    gs = GridSearch(EstimatorAdaptative(metric="f1"), parameters)
    gs.fitAndPredict(X_est, X_pred, y_est, y_pred)

    scores = np.array(gs.results).reshape(len(parameters['alpha']), len(parameters['rho']))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(rho_range, alpha_range)
    Z = np.array(gs.results).reshape(len(alpha_range), len(rho_range))

    print("max_metric" + str(max(gs.results)))

    # Plot the surface.
    ax.set_zlim(0, 1)
    ax.set_title(seq_name)
    ax.set_xlabel('rho')
    ax.set_ylabel('alpha')
    ax.set_zlabel('F1-score')
    colormap = plt.cm.viridis
    normalize = mpl.colors.Normalize(vmin=0, vmax=max(gs.results))
    ax.plot_surface(X, Y, Z, cmap=colormap, norm=normalize)

    plt.show()

