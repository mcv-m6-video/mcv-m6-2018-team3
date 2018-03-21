import sys
import matplotlib.pyplot as plt
from estimator_adaptative import evaluate, EstimatorAdaptative
from sklearn import metrics
from utils import *
from task3 import task3
from task1 import task1
from task2 import task2
from task4 import task4
from scipy import interpolate

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
PlotsDirectory = '../plots/Week3/task5/'
if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

Pr_A, Re_A, Pr_B, Re_B, Pr_sh_A, Re_sh_A, Pr_sh_B, Re_sh_B = list(), list(), list(), list(), list(), list(), list(), list()
Pr_w2, Re_w2, Pr_h4, Re_h4, Pr_t2, Re_t2 = list(), list(), list(), list(), list(), list()
names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
a = [{'min':0, 'max':40, 'step':1}, {'min':0, 'max':40, 'step':1},{'min':0, 'max':40, 'step':1}]
pixels = [4, 16, 5]
rho = [0.599, 0.004,0]

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

        alpha_range = np.arange(a[i].get('min'), a[i].get('max'), a[i].get('step'))

        for idx, alpha in enumerate(alpha_range):
            print(str(idx) + "/" + str(len(alpha_range)) + " " + str(alpha))
            estP_w2 = EstimatorAdaptative(alpha=alpha, rho=rho[i], metric="precision")
            estR_w2 = EstimatorAdaptative(alpha=alpha, rho=rho[i], metric="recall")
            estP_w2.fit(X_est)
            estR_w2.fit(X_est)
            X_res_A = task3(X_est, X_pred, rho[i], alpha, True)
            X_res_B = task3(X_est, X_pred, rho[i], alpha, False)
            X_res_h4,_ = task1(X_est, X_pred, rho[i], alpha, connectivity=4)
            X_res_t2 = task2(X_est, X_pred, rho[i], alpha, pixels[i])
            X_res_sh_A = task4(X_est, X_pred, rho[i], alpha, True)
            X_res_sh_B = task4(X_est, X_pred, rho[i], alpha, False)


            Pr_w2.append(estP_w2.score(y_pred, X=X_pred))
            Re_w2.append(estR_w2.score(y_pred, X=X_pred))
            Pr_h4.append(evaluate(X_res_h4, y_pred, "precision"))
            Re_h4.append(evaluate(X_res_h4, y_pred, "recall"))
            Pr_A.append(evaluate(X_res_A, y_pred, "precision"))
            Re_A.append(evaluate(X_res_A, y_pred, "recall"))
            Pr_B.append(evaluate(X_res_B, y_pred, "precision"))
            Re_B.append(evaluate(X_res_B, y_pred, "recall"))
            Pr_t2.append(evaluate(X_res_t2, y_pred, "precision"))
            Re_t2.append(evaluate(X_res_t2, y_pred, "recall"))
            Pr_sh_A.append(evaluate(X_res_sh_A, y_pred, "precision"))
            Re_sh_A.append(evaluate(X_res_sh_A, y_pred, "recall"))
            Pr_sh_B.append(evaluate(X_res_sh_B, y_pred, "precision"))
            Re_sh_B.append(evaluate(X_res_sh_B, y_pred, "recall"))

        np.save(PlotsDirectory + names[i] +'_Pr_w2.npy', Pr_w2)
        np.save(PlotsDirectory + names[i] +'_Re_w2.npy', Re_w2)
        np.save(PlotsDirectory + names[i] +'_Pr_h4.npy', Pr_h4)
        np.save(PlotsDirectory + names[i] +'_Re_h4.npy', Re_h4)
        np.save(PlotsDirectory + names[i] +'_Pr_A.npy', Pr_A)
        np.save(PlotsDirectory + names[i] +'_Re_A.npy', Re_A)
        np.save(PlotsDirectory + names[i] +'_Pr_B.npy', Pr_B)
        np.save(PlotsDirectory + names[i] +'_Re_B.npy', Re_B)
        np.save(PlotsDirectory + names[i] +'_Pr_t2.npy', Pr_t2)
        np.save(PlotsDirectory + names[i] +'_Re_t2.npy', Re_t2)
        np.save(PlotsDirectory + names[i] +'_Pr_sh_A.npy', Pr_sh_A)
        np.save(PlotsDirectory + names[i] +'_Re_sh_A.npy', Re_sh_A)
        np.save(PlotsDirectory + names[i] +'_Pr_sh_B.npy', Pr_sh_B)
        np.save(PlotsDirectory + names[i] +'_Re_sh_B.npy', Re_sh_B)

        # Empty lists
        Pr_A[:] = []
        Re_A[:] = []
        Pr_B[:] = []
        Re_B[:] = []
        Pr_w2[:] = []
        Re_w2[:] = []
        Pr_h4[:] = []
        Re_h4[:] = []
        Pr_t2[:] = []
        Re_t2[:] = []
        Pr_sh_A[:] = []
        Re_sh_A[:] = []
        Pr_sh_B[:] = []
        Re_sh_B[:] = []

        if len(sys.argv) > 1:
            break

else:
    for i in range(len(names)):
        if len(sys.argv) > 1:
                if len(sys.argv) == 2:
                    i = names.index(str(sys.argv[1]))

        print('plotting ' + names[i] + ' ...')

        Pr_w2 = np.load(PlotsDirectory + names[i] +'_Pr_w2.npy')
        Re_w2 = np.load(PlotsDirectory + names[i] +'_Re_w2.npy')
        Pr_h4 = np.load(PlotsDirectory + names[i] +'_Pr_h4.npy')
        Re_h4 = np.load(PlotsDirectory + names[i] +'_Re_h4.npy')
        Pr_t2 = np.load(PlotsDirectory + names[i] +'_Pr_t2.npy')
        Re_t2 = np.load(PlotsDirectory + names[i] +'_Re_t2.npy')
        Pr_A = np.load(PlotsDirectory + names[i] + '_Pr_A.npy')
        Re_A = np.load(PlotsDirectory + names[i] + '_Re_A.npy')
        Pr_B = np.load(PlotsDirectory + names[i] + '_Pr_B.npy')
        Re_B = np.load(PlotsDirectory + names[i] + '_Re_B.npy')
        Pr_sh_A = np.load(PlotsDirectory + names[i] + '_Pr_sh_A.npy')
        Re_sh_A = np.load(PlotsDirectory + names[i] + '_Re_sh_A.npy')
        #Pr_sh_B = np.load(PlotsDirectory + names[i] + '_Pr_sh_B.npy')
        #Re_sh_B = np.load(PlotsDirectory + names[i] + '_Re_sh_B.npy')

        filled_plot(Pr_w2, Re_w2, Pr_h4, Re_h4, 'week2', '+holefilling', PlotsDirectory, names[i])
        filled_plot(Pr_h4, Re_h4, Pr_t2, Re_t2, 'holefilling', '+opening', PlotsDirectory, names[i])
        filled_plot(Pr_w2, Re_w2, Pr_A, Pr_A, 'week2', '+clossing+holefilling2', PlotsDirectory, names[i])
        filled_plot(Pr_w2, Re_w2, Pr_B, Re_B, 'week2', '+small opening', PlotsDirectory, names[i])
        filled_plot(Pr_w2, Re_w2, Pr_sh_A, Pr_sh_A, 'week2', '+shadow_detection', PlotsDirectory, names[i])

        interp = 500
        x = np.linspace(0, 1.0, interp)
        y3 = np.zeros(interp)
        f1 = interpolate.interp1d(Re_w2, Pr_w2, bounds_error=False)
        f2 = interpolate.interp1d(Re_h4, Pr_h4, bounds_error=False)
        f3 = interpolate.interp1d(Re_B, Pr_B, bounds_error=False)
        f4 = interpolate.interp1d(Re_t2, Pr_t2, bounds_error=False)
        plt.figure()
        line1, = plt.plot(np.array(x), f1(x), 'k',
                          label='week2' + ' = ' + str(round(metrics.auc(Re_w2, Pr_w2, True), 4)))
        line2, = plt.plot(np.array(x), f2(x), color='#005600',
                          label='+holefilling' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        line3, = plt.plot(np.array(x), f3(x), color='#009700',
                          label='+small opening' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        line4, = plt.plot(np.array(x), f4(x), color='#00f300',
                          label='+opening' + ' = ' + str(round(metrics.auc(Re_h4, Pr_h4, True), 4)))
        plt.fill_between(x, f1(x), f2(x), where=f2(x) > f1(x), facecolor='#005600', interpolate=True)
        plt.fill_between(x, f2(x), f3(x), where=f3(x) > f2(x), facecolor='#009700', interpolate=True)
        plt.fill_between(x, f3(x), f4(x), where=f4(x) > f3(x), facecolor='#00f300', interpolate=True)
        plt.fill_between(x, y3, f2(x), where=np.all([f2(x) <= f1(x), f2(x) <= f3(x), f2(x) <= f4(x)], axis=0), facecolor='black', interpolate=True)
        plt.fill_between(x, y3, f1(x), where=np.all([f1(x) <= f2(x), f1(x) <= f3(x), f1(x) <= f4(x)], axis=0), facecolor='black', interpolate=True)
        plt.title("Precision vs Recall curve " + names[i] + " sequence]")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.savefig(PlotsDirectory + names[i] + '_general_PRcurve_AUC.png', bbox_inches='tight')
        plt.close()



