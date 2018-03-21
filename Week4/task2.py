import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from utils import video_to_frame, load_data
from scipy import interpolate

sys.path.append("./backup_week2")
from backup_week2.task2 import task2 as w3task2
from backup_week2.estimator_adaptative import evaluate, EstimatorAdaptative

def filled_plot(Pr1, Re1, Pr2, Re2, name1, name2, directory, dataset):
    max_val = np.max([np.max(Re1), np.max(Re2)])
    interp = 500
    x = np.linspace(0, max_val, interp)
    y3 = np.zeros(interp)
    f1 = interpolate.interp1d(Re1, Pr1,bounds_error=False)
    f2 = interpolate.interp1d(Re2, Pr2,bounds_error=False)
    plt.figure()
    line1, = plt.plot(np.array(x), f1(x), 'k',
                      label=name1+' = ' + str(round(metrics.auc(Re1, Pr1, True), 4)))
    line2, = plt.plot(np.array(x), f2(x), 'g',
                      label=name2+' = ' + str(round(metrics.auc(Re2, Pr2, True), 4)))
    plt.fill_between(x, f1(x), f2(x), where=f2(x) > f1(x), facecolor='green', interpolate=True)
    plt.fill_between(x, f1(x), f2(x), where=f2(x) < f1(x), facecolor='red', interpolate=True)
    plt.fill_between(x, y3, f2(x), where=f2(x) <= f1(x), facecolor='black', interpolate=True)
    plt.fill_between(x, y3, f1(x), where=f1(x) <= f2(x), facecolor='black', interpolate=True)
    plt.title("Precision vs Recall curve " + names[i] + " sequence]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.savefig(directory + dataset +'_'+ name1+'_'+name2+'_PRcurve_AUC.png', bbox_inches='tight')
    plt.close()

    return

data_path = '../../databases'
PlotsDirectory = '../plots/Week4/task2/'
if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

Pr_t2, Re_t2 = [], []
Pr_o, Re_o = [], []
Pr_c, Re_c = [], []
names = ['traffic']
estimation_range = [np.array([950, 1000])]
prediction_range = [np.array([1001, 1050])]
new_estimation_range = [np.array([950 - 950, 1000 - 950])]
new_prediction_range = [np.array([1001 - 950, 1050 - 950])]
a = [{'min':0, 'max':40, 'step':1}]
pixels = [5]
rho = [0]

#Modify this option if you want to compute ROC or PR curves
doComputation = True

if doComputation:
    for i in range(len(names)):
        if len(sys.argv) > 1:
            if len(sys.argv) == 2:
                i = names.index(str(sys.argv[1]))

        print('computing ' + names[i] +' ...')

        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)

        print(X_est.shape)
        print(X_pred.shape)
        print(y_est.shape)
        print(y_pred.shape)

        # vid_o = video_to_frame(data_path+"/traffic/traffic_stabilized_other.mp4")
        # vid_o_gt = video_to_frame(data_path + "/traffic/traffic_stabilized_other_gt.mp4")
        # X_est_o = vid_o[estimation_range[i][0],estimation_range[i][1]]
        # y_est_o = vid_o_gt[estimation_range[i][0],estimation_range[i][1]]
        # X_pred_o = vid_o[prediction_range[i][0], prediction_range[i][1]]
        # y_pred_o = vid_o_gt[prediction_range[i][0], prediction_range[i][1]]

        vid_c = np.load(data_path + "/traffic/custom/traffic_stabilized_bloque5_area10.npy")
        vid_c_gt = np.load(data_path + "/traffic/custom/traffic_gt_stabilized_bloque5_area10.npy")
        X_est_c = vid_c[new_estimation_range[i][0]: new_estimation_range[i][1]]
        y_est_c = vid_c_gt[new_estimation_range[i][0]:new_estimation_range[i][1]]
        X_pred_c = vid_c[new_prediction_range[i][0]:new_prediction_range[i][1]]
        y_pred_c = vid_c_gt[new_prediction_range[i][0]: new_prediction_range[i][1]]

        X_est_o =  X_est_c
        y_est_o = y_est_c
        X_pred_o = X_pred_c
        y_pred_o = y_pred_c

        alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

        for idx, alpha in enumerate(alpha_range):
            print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
            X_res_t2 = w3task2(X_est, X_pred, rho[i], alpha, pixels[i])
            X_res_o = w3task2(X_est_o, X_pred_o, rho[i], alpha, pixels[i])
            X_res_c = w3task2(X_est_c, X_pred_c, rho[i], alpha, pixels[i])

            Pr_t2.append(evaluate(X_res_t2, y_pred, "precision"))
            Re_t2.append(evaluate(X_res_t2, y_pred, "recall"))
            Pr_o.append(evaluate(X_res_o, y_pred, "precision"))
            Re_o.append(evaluate(X_res_o, y_pred, "recall"))
            Pr_c.append(evaluate(X_res_c, y_pred, "precision"))
            Re_c.append(evaluate(X_res_c, y_pred, "recall"))

        np.save(PlotsDirectory + names[i] +'_Pr_t2.npy', Pr_t2)
        np.save(PlotsDirectory + names[i] +'_Re_t2.npy', Re_t2)
        np.save(PlotsDirectory + names[i] + '_Pr_o.npy', Pr_o)
        np.save(PlotsDirectory + names[i] + '_Re_o.npy', Re_o)
        np.save(PlotsDirectory + names[i] + '_Pr_c.npy', Pr_c)
        np.save(PlotsDirectory + names[i] + '_Re_c.npy', Re_c)

        # Empty lists
        Pr_t2[:] = []
        Re_t2[:] = []
        Pr_o[:] = []
        Re_o[:] = []
        Pr_c[:] = []
        Re_c[:] = []

        if len(sys.argv) > 1:
            break

else:
    for i in range(len(names)):
        if len(sys.argv) > 1:
                if len(sys.argv) == 2:
                    i = names.index(str(sys.argv[1]))

        print('plotting ' + names[i] + ' ...')

        Pr_t2 = np.load(PlotsDirectory + names[i] +'_Pr_t2.npy')
        Re_t2 = np.load(PlotsDirectory + names[i] +'_Re_t2.npy')
        Pr_c = np.load(PlotsDirectory + names[i] + '_Pr_c.npy')
        Re_c = np.load(PlotsDirectory + names[i] + '_Re_c.npy')
        Pr_o = np.load(PlotsDirectory + names[i] + '_Pr_o.npy')
        Re_o = np.load(PlotsDirectory + names[i] + '_Re_o.npy')

        filled_plot(Pr_t2, Re_t2, Pr_c, Re_c, 'best week3', 'custom stablilization', PlotsDirectory, names[i])
        filled_plot(Pr_t2, Re_t2, Pr_o, Re_o, 'best week3', 'other stablilization', PlotsDirectory, names[i])


        # Todo Unified plot....
        # interp = 500
        # x = np.linspace(0, 1.0, interp)
        # y3 = np.zeros(interp)
        # f1 = interpolate.interp1d(Re_w2, Pr_w2, bounds_error=False)
        # f2 = interpolate.interp1d(Re_h4, Pr_h4, bounds_error=False)
        # f3 = interpolate.interp1d(Re_B, Pr_B, bounds_error=False)
        # f4 = interpolate.interp1d(Re_t2, Pr_t2, bounds_error=False)
        # plt.figure()
        # line1, = plt.plot(np.array(x), f1(x), 'k',
        #                   label='week2' + ' = ' + str(round(metrics.auc(Re_w2, Pr_w2, True), 4)))
        # line2, = plt.plot(np.array(x), f2(x), color='#005600',
        #                   label='+holefilling' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        # line3, = plt.plot(np.array(x), f3(x), color='#009700',
        #                   label='+small opening' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        # line4, = plt.plot(np.array(x), f4(x), color='#00f300',
        #                   label='+opening' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        # plt.fill_between(x, f1(x), f2(x), where=f2(x) > f1(x), facecolor='#005600', interpolate=True)
        # plt.fill_between(x, f2(x), f3(x), where=f3(x) > f2(x), facecolor='#009700', interpolate=True)
        # plt.fill_between(x, f3(x), f4(x), where=f4(x) > f3(x), facecolor='#00f300', interpolate=True)
        # plt.fill_between(x, y3, f2(x), where=np.all([f2(x) <= f1(x), f2(x) <= f3(x), f2(x) <= f4(x)], axis=0), facecolor='black', interpolate=True)
        # plt.fill_between(x, y3, f1(x), where=np.all([f1(x) <= f2(x), f1(x) <= f3(x), f1(x) <= f4(x)], axis=0), facecolor='black', interpolate=True)
        # plt.title("Precision vs Recall curve " + names[i] + " sequence]")
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.5, -0.1))
        # plt.savefig(PlotsDirectory + names[i] + '_general_PRcurve_AUC.png', bbox_inches='tight')
        # plt.close()

