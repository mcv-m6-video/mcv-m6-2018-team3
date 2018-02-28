import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def load_data(data_path, seq_range=None):

    X = []
    y = []

    if seq_range is None: seq_range = [0, len(glob.glob(data_path + 'groundtruth/*'))]

    for index in range(seq_range[0], seq_range[1]+1):
        in_name = data_path + 'input/in' + str(index).zfill(6) + '.jpg'
        gt_name = data_path + 'groundtruth/gt' + str(index).zfill(6) + '.png'

        in_image = cv2.imread(in_name)
        gt_image = cv2.imread(gt_name)

        X.append(in_image)
        y.append(gt_image)


    return np.array(X), np.array(y)

def pixel_evaluation(ground_truth, prediction):
    assert len(ground_truth.shape) == 3 and len(prediction.shape) == 3

    ground_truth = np.array(ground_truth[:,:,0])
    prediction = np.array(prediction[:,:,0])

    TP = len(np.where(ground_truth[np.where(prediction == 1)] == 255)[0])
    FP = len(np.where(ground_truth[np.where(prediction == 1)] != 255)[0])

    FN = len(np.where(ground_truth[np.where(prediction == 0)] == 255)[0])
    TN = len(np.where(ground_truth[np.where(prediction == 0)] != 255)[0])

    TF = len(np.where(ground_truth == 255)[0])

    return np.array([TP, TN, FP, FN, TF])

def precision(pe):
    return pe[0]/(pe[0]+pe[2])

def recall(pe):
    return pe[0]/(pe[0]+pe[3])

def f1_score(pe):
    return 2*pe[0]/(2*pe[0]+pe[2]+pe[3])

def visual_of(im, gtx, gty, gtz, overlap=0.9, wsize=300, mult=1, thickness=1):
    step = int(wsize * (1 - overlap))
    mwsize = int(wsize / 2)
    h,w = gtx.shape

    for i in np.arange(-mwsize,h+1-mwsize,step):
        for j in np.arange(-mwsize,w+1-mwsize,step):
            ai,bi, aj, bj = getCoords(i, j, wsize, h, w)
            mask = gtz[ai:bi, aj:bj]
            if np.count_nonzero(mask) == 0:
                continue
            winx = gtx[ai:bi, aj:bj]
            winy = gty[ai:bi, aj:bj]
            glob_x = (np.sum(winx[mask])*mwsize)/(np.count_nonzero(mask)*512)*mult
            glob_y = (np.sum(winy[mask])*mwsize)/(np.count_nonzero(mask)*512)*mult
            pt1 = (int(j + mwsize), int(i + wsize / 2))
            pt2 = (int(j + mwsize + glob_x), int(i + mwsize + glob_y))
            color = (0, 255, 0)
            im = cv2.arrowedLine(im, pt1, pt2, color, thickness)
    return im

def getCoords(i,j,w_size,h,w):
    if i<0:
        ai=0
    else:
        ai=i

    if j<0:
        aj=0
    else:
        aj=j

    if i+w_size>=h:
        bi=h-1
    else:
        bi=i+w_size

    if j+w_size>=h:
        bj=w-1
    else:
        bj=j+w_size

    return ai, bi, aj, bj
