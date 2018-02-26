import cv2
import numpy as np

def load_data(data_path, seq_range=None):
    """
    load database and return frames between seq_range with numpy arrays
    """
    n_frames = seq_range[1] - seq_range[0]



    return

def pixel_evaluation(ground_truth, prediction):
    ground_truth = ground_truth[0]

    TP = np.count_nonzero(ground_truth * prediction)



