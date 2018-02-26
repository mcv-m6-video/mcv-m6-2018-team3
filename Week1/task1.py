import glob
import re
from collections import OrderedDict

import cv2
import numpy as np
from tabulate import tabulate

from utils import load_data, pixel_evaluation
from utils import precision, recall, f1_score

TestDirectory = '../test_results/foreground/highway/'
GTDirectory = '../databases/highway/'
seq_range = [1201, 1400]

_, gt = load_data(GTDirectory, seq_range)

# test A results
regex = re.compile(".*(test_A).*")
a_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]

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
a_precision = precision(a_pe)
a_recall = recall(a_pe)
a_f1_score = f1_score(a_pe)

# test B results
regex = re.compile(".*(test_B).*")
b_names = [m.group(0) for l in glob.glob(TestDirectory + '*') for m in [regex.search(l)] if m]

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
b_precision = precision(b_pe)
b_recall = recall(b_pe)
b_f1_score = f1_score(b_pe)

# print results
table_dict = OrderedDict()
table_dict['Test'] = ["A", "B"]
table_dict['Precision'] = [a_precision, b_precision]
table_dict['Recall'] = [a_recall, b_recall]
table_dict['F1 Score'] = [a_f1_score, b_f1_score]
print(tabulate(table_dict, headers="keys", tablefmt="fancy_grid"))
