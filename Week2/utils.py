import glob
import os
import cv2
import numpy as np


def load_data(data_path, data_id, seq_range=None, grayscale=True):
    X = []
    y = []

    path = os.path.join(data_path, data_id)

    if seq_range is None: seq_range = [0, len(glob.glob(path + '/groundtruth/*'))]

    for index in range(seq_range[0], seq_range[1] + 1):
        in_name = path + '/input/in' + str(index).zfill(6) + '.jpg'
        gt_name = path + '/groundtruth/gt' + str(index).zfill(6) + '.png'

        if grayscale:
            in_image = cv2.imread(in_name, 0)
            gt_image = cv2.imread(gt_name, 0)
        else:
            in_image = cv2.imread(in_name)
            gt_image = cv2.imread(gt_name)

        X.append(in_image)
        y.append(gt_image)

    return np.array(X), np.array(y)


def fit(X, y):
    idx = np.ones(y.shape) * np.nan

    idx[np.where(y == 0)] = 1
    idx[np.where(y == 50)] = 1

    mean_map = np.nanmean(X * idx, axis=0)
    var_map = np.nanvar(X * idx, axis=0)

    return np.array([mean_map, var_map])

def predict(X, background_model):



def pixel_evaluation(ground_truth, prediction):
    assert len(ground_truth.shape) == 3 and len(prediction.shape) == 3

    ground_truth = np.array(ground_truth[:, :, 0])
    prediction = np.array(prediction[:, :, 0])

    TP = len(np.where(ground_truth[np.where(prediction == 1)] == 255)[0])
    FP = len(np.where(ground_truth[np.where(prediction == 1)] != 255)[0])

    FN = len(np.where(ground_truth[np.where(prediction == 0)] == 255)[0])
    TN = len(np.where(ground_truth[np.where(prediction == 0)] != 255)[0])

    TF = len(np.where(ground_truth == 255)[0])

    return np.array([TP, TN, FP, FN, TF])


def precision(pe):
    return pe[0] / (pe[0] + pe[2])


def recall(pe):
    return pe[0] / (pe[0] + pe[3])


def f1_score(pe):
    return 2 * pe[0] / (2 * pe[0] + pe[2] + pe[3])
