from estimator_adaptative import EstimatorAdaptative
from mpl_toolkits.mplot3d import Axes3D
from grid_search import GridSearch
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

for seq_index, seq_name in enumerate(names):

    print('computing ' + seq_name +' ...')

    [X_est, y_est] = load_data(data_path, seq_name, estimation_range[seq_index], grayscale=True)
    [X_pred, y_pred] = load_data(data_path, seq_name, prediction_range[seq_index], grayscale=True)

    alpha_range = np.arange(0,11)
    rho_range = np.arange(1,10)/10

    parameters = {'alpha': alpha_range, 'rho': rho_range}
    gs = GridSearch(EstimatorAdaptative(metric="f1"), parameters)
    gs.fitAndPredict(X_est, X_pred, y_est, y_pred)

    print('best_metric: '+str(gs.best_score))
    print('best_params: '+str(gs.best_params))
    scores = np.array(gs.results).reshape(len(parameters['alpha']), len(parameters['rho']))


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(rho_range, alpha_range)
    Z = np.array(gs.results).reshape(len(alpha_range), len(rho_range))

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

