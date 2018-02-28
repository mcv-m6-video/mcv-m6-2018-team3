import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import load_data, pixel_evaluation, f1_score

TestDirectory = '../test_results/foreground/highway/'
GTDirectory = '../databases/highway/'
PlotsDirectory = 'Week1/plots/task4/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

n_syncs = 200
a_desync_F1 = np.zeros(n_syncs)
b_desync_F1 = np.zeros(n_syncs)

for sy_index in range(0, n_syncs):

    print(sy_index)
    seq_range = np.array([1201, 1400]) + sy_index
    _, gt = load_data(GTDirectory, seq_range)

    # test A results
    regex = re.compile(".*(test_A).*")
    a_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]
    a_names.sort()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, name in enumerate(a_names):
        prediction = cv2.imread(name)
        ground_truth = gt[index]

        pe = pixel_evaluation(ground_truth, prediction)

        TP += pe[0]
        TN += pe[1]
        FP += pe[2]
        FN += pe[3]

    a_pe = np.array([TP, TN, FP, FN])
    a_desync_F1[sy_index] = f1_score(a_pe)

    # test B results
    regex = re.compile(".*(test_B).*")
    b_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]
    b_names.sort()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, name in enumerate(b_names):
        prediction = cv2.imread(name)
        ground_truth = gt[index]

        pe = pixel_evaluation(ground_truth, prediction)

        TP += pe[0]
        TN += pe[1]
        FP += pe[2]
        FN += pe[3]

    b_pe = np.array([TP, TN, FP, FN])
    b_desync_F1[sy_index] = f1_score(b_pe)

# plot results
frames = np.arange(n_syncs)

plt.figure(1)
a_line1, = plt.plot(frames, a_desync_F1, 'b', label='Test A')
a_line2, = plt.plot(frames, b_desync_F1, 'r', label='Test B')
plt.title("De-syncronized F1")
plt.xlabel("frame")
plt.legend(handles=[a_line1, a_line2], loc='upper center', bbox_to_anchor=(0.5, -0.1))

plt.savefig(PlotsDirectory + 'de-syncronized-F1 _plot.png', bbox_inches='tight')

plt.show()
