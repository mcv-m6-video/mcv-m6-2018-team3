import glob
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import f1_score
from utils import load_data, pixel_evaluation

TestDirectory = '../test_results/foreground/highway/'
GTDirectory = '../databases/highway/'
seq_range = [1201, 1400]
n_frames = seq_range[1] - seq_range[0] + 1

_, gt = load_data(GTDirectory, seq_range)

# test A results
regex = re.compile(".*(test_A).*")
a_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]

a_TP = np.zeros(n_frames)
a_TF = np.zeros(n_frames)
a_F1 = np.zeros(n_frames)

for index, name in enumerate(a_names):
    prediction = cv2.imread(name)
    ground_truth = gt[index]

    pe = pixel_evaluation(ground_truth, prediction)

    a_TP[index] = pe[0]
    a_TF[index] = pe[4]
    a_F1[index] = f1_score(pe)

# test B results
regex = re.compile(".*(test_B).*")
b_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]

b_TP = np.zeros(n_frames)
b_TF = np.zeros(n_frames)
b_F1 = np.zeros(n_frames)

for index, name in enumerate(b_names):
    prediction = cv2.imread(name)
    ground_truth = gt[index]

    pe = pixel_evaluation(ground_truth, prediction)

    b_TP[index] = pe[0]
    b_TF[index] = pe[4]
    b_F1[index] = f1_score(pe)

# plot results
frames = np.arange(n_frames)

plt.figure(1)
a_line1, = plt.plot(frames, a_TP, 'b', label = 'True Positives')
a_line2, = plt.plot(frames, a_TF, 'r', label = 'Total Foreground')
plt.title("Test A")
plt.legend(handles=[a_line1, a_line2], loc='upper center', bbox_to_anchor=(0.5,-0.1))

plt.savefig('testA_TP_TF_plot.png', bbox_inches='tight')

plt.figure(2)
b_line1, = plt.plot(frames, b_TP, 'b', label = 'True Positives')
b_line2, = plt.plot(frames, b_TF, 'r', label = 'Total Foreground')
plt.title("Test B")
plt.legend(handles=[b_line1, b_line2], loc='upper center', bbox_to_anchor=(0.5,-0.1))

plt.savefig('testB_TP_TF_plot.png', bbox_inches='tight')

plt.figure(3)
line1, = plt.plot(frames, a_F1, 'b', label = 'Test A')
line2, = plt.plot(frames, b_F1, 'r', label = 'Test B')
plt.title("F1 score")
plt.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5,-0.1))

plt.savefig('F1_plot.png', bbox_inches='tight')