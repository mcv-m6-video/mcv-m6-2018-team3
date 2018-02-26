import cv2
import numpy as np

def load_data(data_path, seq_range=None):
    """
    load database and return frames between seq_range with numpy arrays
    """
    n_frames = seq_range[1] - seq_range[0]

    for index in range(seq_range[0], seq_range[1]+1):
        in_name = data_path + '/input/in' + str(index).zfill(6) + '.jpg'
        gt_name = data_path + '/groundtruth/gt' + str(index).zfill(6) + '.png'

        #TODO: read filenames from data_path



    return

def pixel_evaluation(ground_truth, prediction):
    ground_truth = np.array(ground_truth[0], dtype=bool)
    prediction = np.array(prediction[0], dtype=bool)

    TP = np.count_nonzero(np.logical_and(ground_truth, prediction))
    TN = np.count_nonzero(np.logical_and(np.logical_not(ground_truth), np.logical_not(prediction)))
    FP = np.count_nonzero(np.logical_and(np.logical_not(ground_truth), prediction))
    FN = np.count_nonzero(np.logical_and(ground_truth, np.logical_not(prediction)))

    return np.array([TP, TN, FP, FN])

def precision(pe):
    return pe[0]/(pe[0]+pe[2])

def recall(pe):
    return pe[0]/(pe[0]+pe[3])

def f1_score(pe):
    return 2*pe[0]/(2*pe[0]+pe[2]+pe[3])