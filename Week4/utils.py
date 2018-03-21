import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def video_to_frame(filename, grayscale=True):
    vidcap = cv2.VideoCapture(filename)
    # Check if camera opened successfully
    if vidcap.isOpened() is False:
        print("Error opening video stream or file")
    frames_vol=[]
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if type(frame) == type(None):
            break
        if grayscale: frame= rgb2gray(frame)
        frames_vol.append(frame)
    frames_vol=np.np.array(frames_vol)

    return frames_vol

def video_to_frame_other(filename):
    vidcap = cv2.VideoCapture('filename')
    # Check if camera opened successfully
    if vidcap.isOpened() is False:
        print("Error opening video stream or file")
    frames_vol=[]
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        frame1= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_vol.append(frame1)
    frames_vol=np.array(frames_vol)
    s=frames_vol.shape

    return frames_vol, s

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
            gt_image = cv2.imread(gt_name, 0)

        X.append(in_image)
        y.append(gt_image)

    return np.array(X), np.array(y)


# INPUT: X: is a sequence of images, path: directory to save images.
def write_images(X, path, head_filename):

    path = os.path.join(path, head_filename)

    for i in range(X.shape[0]):
        plt.figure()
        filename = path + str(i).zfill(6) + '.png'
        plt.imshow(X[i], cmap="gray")
        plt.savefig(filename)
        plt.close()

    return

def write_images2(X, path, head_filename):

    path = os.path.join(path, head_filename)

    for i in range(X.shape[0]):
        filename = path + str(i).zfill(6) + '.png'
        cv2.imwrite(filename, X[i]);
    return

def simplify_labels(y):
    aux = np.ones(y.shape) * np.nan
    aux[np.where(y == 0)] = 1
    aux[np.where(y == 50)] = 1
    return aux

def build_mask(y):
    # Convert ground truth to mask
    mask = np.ones(y.shape)
    mask[np.where(y == 0)] = 0
    mask[np.where(y == 50)] = 0
    mask[np.where(y == 85)] = np.nan
    mask[np.where(y == 170)] = np.nan

    return mask

def fit(X, y):
    idx = simplify_labels(y)
    mean_map = np.nanmean(X * idx, axis=0)
    var_map = np.nanvar(X * idx, axis=0)

    return np.array([mean_map, var_map])

def predict(X, background_model, alpha):
    mean_map = background_model[0]
    var_map = background_model[1]

    prediction = np.zeros(X.shape)
    prediction[np.absolute(X - mean_map) >= alpha * (var_map + 2)] = 1

    return prediction


def pixel_evaluation(predictions, ground_truth):


    # ground_truth: first call build_mask
    idx = np.where(~ np.isnan(ground_truth))
    ground_truth = ground_truth[idx]
    predictions = predictions[idx]


    TP = len(np.where(ground_truth[np.where(predictions == 1)] == 1)[0])
    FP = len(np.where(ground_truth[np.where(predictions == 1)] == 0)[0])

    FN = len(np.where(ground_truth[np.where(predictions == 0)] == 1)[0])
    TN = len(np.where(ground_truth[np.where(predictions == 0)] == 0)[0])

    TF = len(np.where(ground_truth == 1)[0])

    return np.array([TP, TN, FP, FN, TF])


def precision(pe):
    return pe[0] / (pe[0] + pe[2])


def recall(pe):
    return pe[0] / (pe[0] + pe[3])


def f1_score(pe):
    return 2 * pe[0] / (2 * pe[0] + pe[2] + pe[3])

def fpr_metric(pe):
    return pe[2]/(pe[2]+pe[1])

def tpr_metric(pe):
    return pe[0] / (pe[0] + pe[3])

def write_video(sequence, output_path):
    height, width = sequence[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=25 ,frameSize=(height, width), isColor=0)
    for frame in sequence:
        print(frame.shape)
        video.write(frame)
        video.release()

def MOG2(X_pred):

    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2()

    shadowMOG = np.zeros(X_pred.shape)
    for idx, frame in enumerate(X_pred):

        shadow=fgbgMOG2.apply(frame)
        shadowMOG[idx][shadow == 127] = 1

    return shadowMOG
