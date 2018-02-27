import cv2
import numpy as np
import matplotlib.pyplot as plt

#First pair
gt45_name = "../databases/data_stereo_flow/training/flow_noc/000045_10.png"
test45_name = "../test_results/motion/LKflow_000045_10.png"

gt45 = cv2.imread(gt45_name, -1),

test45 = cv2.imread(test45_name, -1)

gt45x = (np.array(gt45[:,:,0], dtype=float)-(2**15))/64.0
gt45y = (np.array(gt45[:,:,1], dtype=float)-(2**15))/64.0
gt45z = np.array(gt45[:,:,2],dtype=bool)
print(len((np.where(gt45z==True)[0])))

test45x = (np.array(test45[:,:,0], dtype=float)-(2**15))/64.0
test45y = (np.array(test45[:,:,1], dtype=float)-(2**15))/64.0
test45z = np.array(test45[:,:,2],dtype=bool)
print(len((np.where(test45z==True)[0])))
mask1 = np.logical_and(gt45z,test45z)

gt45x_1 = gt45x*mask1
gt45y_1 = gt45y*mask1
test45x_1 = test45x*mask1
test45y_1 = test45y*mask1

#Mean Square error in Non-ocluded areas
msen45 = np.sqrt((gt45x_1-test45x_1)**2+(gt45y_1-test45y_1)**2)
print(len(np.where(mask1==True)[0]))
#plt.hist(msen45)
plt.show()
m_msen45 = np.mean(np.mean(msen45))
print(m_msen45)

modul_gt45 = np.sqrt(gt45x_1**2+gt45y_1**2)
modul_test45 = np.sqrt(test45x_1**2+test45y_1**2)
mask2 = np.logical_and(modul_gt45>3.0,modul_test45>3.0)

gt45x_2 = gt45x*mask2
gt45y_2 = gt45y*mask2
test45x_2 = test45x*mask2
test45y_2 = test45y*mask2

#Percentage of Erroneous Pixels in Non-occluded areas
erroneous45 = np.logical_and(mask1,mask2)
pepn45 = np.count_nonzero(erroneous45)/np.count_nonzero(mask1)
print(pepn45)

#Second pair
gt157_name = "../databases/data_stereo_flow/training/flow_noc/0000157_10.png"
test157_name = "../test_results/motion/LKflow_000157_10.png"

gt157 = cv2.imread(gt157_name)
test157 = cv2.imread(test157_name)
                    
gt157x = (np.array(gt157[:,:,0], dtype=float)-(2**15))/64.0
gt157y = (np.array(gt157[:,:,1], dtype=float)-(2**15))/64.0
gt157z = np.array(gt157[:,:,2],dtype=bool)

test157x = (np.array(test157[:,:,0], dtype=float)-(2**15))/64.0
test157y = (np.array(test157[:,:,1], dtype=float)-(2**15))/64.0
test157z = np.array(test157[:,:,2],dtype=bool)

mask1 = np.logical_and(gt157z,test157z)

gt157x_1 = gt157x*mask1
gt157y_1 = gt157y*mask1
test157x_1 = test157x*mask1
test157y_1 = test157y*mask1

#Mean Square error in Non-ocluded areas
msen157 = np.sqrt((gt157x_1-test157x_1)**2+(gt157y_1-test157y_1)**2)
m_msen157 = np.mean(np.mean(msen157))
print(m_msen157)

modul_gt = np.sqrt(gt157x_1**2+gt157y_1**2)
modul_test = np.sqrt(test157x_1**2+test157y_1**2)
mask2 = np.logical_and(modul_gt>3.0,modul_test>3.0)

gt157x_2 = gt157x*mask2
gt157y_2 = gt157y*mask2
test157x_2 = test157x*mask2
test157y_2 = test157y*mask2

#Percentage of Erroneous Pixels in Non-occluded areas
erroneous = np.logical_and(mask1,mask2)
pepn157 = np.count_nonzero(erroneous)/np.count_nonzero(erroneous45)/np.count_nonzero(mask1)
print(pepn157)
