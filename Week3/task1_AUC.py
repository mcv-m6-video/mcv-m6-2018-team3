import sys
from utils import *
from task1 import *
import matplotlib.pyplot as plt
from sklearn import metrics

Pr = list()
Re = list()
names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
a = [{'min':0, 'max':40, 'step':1}, {'min':0, 'max':40, 'step':1},{'min':0, 'max':40, 'step':1}]
rho = [0.599, 0.004,0]
for i in range(len(names)):
    if len(sys.argv) > 1:
        i = names.index(str(sys.argv[1]))

    alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

    for idx, alpha in enumerate(alpha_range):
        print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))


    plt.figure()
    plt.plot(np.array(Re), np.array(Pr), 'b', label='Precision-Recall')
    plt.title("Precision vs Recall curve [AUC = " + str(round(metrics.auc(Re, Pr, True), 4)) + "] [" + names[i] + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()