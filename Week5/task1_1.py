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
est_range = [np.array([1050, 1200]), np.array([950, 1000])]
pixels = [4, 7] #best kernel dimension for the opening per dataset
alpha = [2, 2.449] #best alpha per dataset (adaptative model)
rho = [0.0759, 0.178] #best rho per dataset (adaptative model)

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
    [X_est, _] = load_data(data_path, names[i], est_range[i], grayscale=True)
    #

    if names[i] == 'highway':
        X_res = w3task2(X_est, X_track, rho[i], alpha[i], pixels[i], 4, 4, True)
    elif names[i] == 'traffic':
        X_res = w3task2(X_est, X_track, rho[i], alpha[i], pixels[i], 8, 8, True)


    # PREPROCESSING
    dataset = names[i]
    if (dataset == 'traffic'):
        print("Making preprocessing for traffic...")

        count=0
        for image, mask in zip(X_color[:, :, :], X_res[:, :, :]):
            nb_objects, cc_map, bboxes, centroids = getConnectedComponents(mask)

            print("count = ", count)
            count += 1

            centroids = np.array(centroids).astype("int")

            for idx in np.unique(cc_map)[1:]:
                area = bboxes[idx][-1:]

                print("area = ", area)

                if 500 < area:
                    for c in np.unique(cc_map)[1:]:
                        #centroids[idx] = np.array(centroids[idx]).astype("int")
                        #centroids[c] = np.array(centroids[c]).astype("int")
                        D = computeDistance(centroids[idx], centroids[c])
                        print(D)
                        A = bboxes[c][-1:]
                        print("A = ", A)
                        if D < 180 and 60 < A:
                            # Draw a diagonal blue line with thickness of 5 px
                            color = (1,1,1)
                            cv2.line(mask, (centroids[idx][0], centroids[idx][1]), (centroids[c][0], centroids[c][1]), color=(1,1,1), thickness=3)

    # elif (dataset == 'highway'):
    #     print("Making preprocessing for highway...")
    #
    #     #count=0
    #     kernel = np.ones((3, 3), np.uint8)
    #
    #     for index, (image, mask) in enumerate(zip(X_color[:, :, :], X_res[:, :, :])):
    #         nb_objects, cc_map, bboxes, centroids = getConnectedComponents(mask)
    #
    #         #print("count = ", count)
    #         #count += 1
    #
    #         centroids = np.array(centroids).astype("int")
    #
    #         for idx in np.unique(cc_map)[1:]:
    #             area = bboxes[idx][-1:]
    #
    #             #print("area = ", area)
    #
    #             # if 350 < area:
    #             #     for c in np.unique(cc_map)[1:]:
    #             #         #centroids[idx] = np.array(centroids[idx]).astype("int")
    #             #         #centroids[c] = np.array(centroids[c]).astype("int")
    #             #         D = computeDistance(centroids[idx], centroids[c])
    #             #         A = bboxes[c][-1:]
    #             #         if D < 45 and 80 < A and A < 300:
    #             #             # Draw a diagonal blue line with thickness of 5 px
    #             #             color = (1,1,1)
    #             #             cv2.line(mask, (centroids[idx][0], centroids[idx][1]), (centroids[c][0], centroids[c][1]), color=(1,1,1), thickness=3)
    #                         #print("distance = ", D)
    #
    #         mask = cv2.dilate(mask * 255, kernel, iterations=1)
    #
    #         mask = mask.astype("int")
    #         mask[np.where(mask != 0)] = 1
    #         X_res[index] = mask


    # finally save the masks necessary to process with kalman filter or other filter
    np.save('masks_new.npy', X_res)
    write_images2(X_res * 255, 'output', 'mask_')

    #Tracking = kalmanFilter(X_res) #Todo kalamn filter function


    if len(sys.argv) > 1: #Execute one time if dataset is selected (from command line)
        break
