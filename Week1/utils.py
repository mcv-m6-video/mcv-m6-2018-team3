import cv2
import numpy as np

def load_data(data_path, seq_range=None):
    """
    load database and return frames between seq_range with numpy arrays
    """
    n_frames = seq_range[1] - seq_range[0]


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