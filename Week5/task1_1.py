import sys
import os
import numpy as np
from utils import load_data, write_images2
import cv2

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

def computeDistance(point1, point2):
    distance = pow((point1[0] - point2[0])** 2 + (point1[1] - point2[1])** 2, 0.5)
    return distance

def getConnectedComponents(mask):
    connectivity = 4
    mask = mask * 255.
    mask = mask.astype("uint8")

    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    nb_objects = output[0] - 1
    cc_map = output[1]
    bboxes = output[2]
    centroids = output[3]

    return nb_objects, cc_map, bboxes, centroids

for i in range(len(names)):
    if len(sys.argv) > 1: # Execute only the dataset selected (from command line)
        if len(sys.argv) == 2:
            i = names.index(str(sys.argv[1]))

    print('computing ' + names[i] +' ...')

    [X_color, _] = load_data(data_path, names[i], tracking_range[i], grayscale=False)
    np.save('original_images.npy', X_color)
    write_images2(X_color, 'output', 'img_')

    [X_track, _ ] = load_data(data_path, names[i], tracking_range[i], grayscale=True)
    X_res = w3task2(X_track, X_track, rho[i], alpha[i], pixels[i])

    # PREPROCESSING
    count=0
    for image, mask in zip(X_color[:, :, :], X_res[:, :, :]):
        nb_objects, cc_map, bboxes, centroids = getConnectedComponents(mask)

        print("count = ", count)
        count += 1

        centroids = np.array(centroids).astype("int")

        for idx in np.unique(cc_map)[1:]:
            area = bboxes[idx][-1:]

            print("area = ", area)

            if 1500 < area:
                for c in np.unique(cc_map)[1:]:
                    #centroids[idx] = np.array(centroids[idx]).astype("int")
                    #centroids[c] = np.array(centroids[c]).astype("int")
                    D = computeDistance(centroids[idx], centroids[c])
                    print(D)
                    A = bboxes[c][-1:]
                    if D < 90 and A > 45:
                        # Draw a diagonal blue line with thickness of 5 px
                        color = (1,1,1)
                        cv2.line(mask, (centroids[idx][0], centroids[idx][1]), (centroids[c][0], centroids[c][1]), color=(1,1,1), thickness=3)


    np.save('masks.npy', X_res)
    write_images2(X_res*255, 'output', 'mask_')
    #Tracking = kalmanFilter(X_res) #Todo kalamn filter function


    if len(sys.argv) > 1: #Execute one time if dataset is selected (from command line)
        break
