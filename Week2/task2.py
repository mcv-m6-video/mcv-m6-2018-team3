from estimator_adaptative import EstimatorAdaptative
from grid_search import GridSearch
import matplotlib.pyplot as plt
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

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=True)

    parameters = {'alpha': 3+np.arange(7), 'rho': 1/np.arange(2,5)}
    gs = GridSearch(EstimatorAdaptative(metric="f1"), parameters)
    gs.fit(X_est, y=y_est)


    scores = np.array(gs.cv_results_['mean_test_score']).reshape(len(parameters['alpha']), len(parameters['rho']))


    for i, alpha in enumerate(parameters['alpha']):
        plt.plot(parameters['rho'], scores[i], label='alpha: ' + str(alpha))

    plt.legend()
    plt.xlabel('rho')
    plt.ylabel('f1 score')
    plt.show()

    print(gs.best_params_)
    print(gs.best_score_)
    print(gs.cv_results_['mean_train_score'])
