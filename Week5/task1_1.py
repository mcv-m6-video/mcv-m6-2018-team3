import sys
import os
import numpy as np
from utils import load_data

sys.path.append("./backup_week2")
from backup_week2.task2 import task2 as w3task2

#Creation of directories to save data
data_path = '../../databases'
PlotsDirectory = '../plots/Week5/task1_1/'
if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

#Definition of variables
names = ['highway', 'traffic']
tracking_range = [np.array([1050, 1350]), np.array([950, 1050])]
pixels = [4, 5] #best kernel dimension for the opening per dataset
alpha = [7.25, 10.67] #best alpha per dataset (adaptative model)
rho = [0.599, 0] #best rho per dataset (adaptative model)


for i in range(len(names)):
    if len(sys.argv) > 1: # Execute only the dataset selected (from command line)
        if len(sys.argv) == 2:
            i = names.index(str(sys.argv[1]))

    print('computing ' + names[i] +' ...')

    [X_track, _ ] = load_data(data_path, names[i], tracking_range[i], grayscale=True)

    X_res = w3task2(X_track, X_track, rho[i], alpha[i], pixels[i])
    #Tracking = kalmanFilter(X_res) #Todo kalamn filter function




    if len(sys.argv) > 1: #Execute one time if dataset is selected (from command line)
        break

