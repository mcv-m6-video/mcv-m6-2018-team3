import cv2
import matplotlib.pyplot as plt
import numpy as np

# First pair
gt45_name = "../databases/data_stereo_flow/training/flow_noc/000045_10.png"
test45_name = "../test_results/motion/LKflow_000045_10.png"

gt45 = cv2.imread(gt45_name, -1)
test45 = cv2.imread(test45_name, -1)

gt45x = (np.array(gt45[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
gt45y = (np.array(gt45[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
gt45z = np.array(gt45[:, :, 0], dtype=bool)
print("non ocluded gt:" + str(np.count_nonzero(gt45z)))

test45x = (np.array(test45[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
test45y = (np.array(test45[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
test45z = np.array(test45[:, :, 0], dtype=bool)
print("non ocluded test:" + str(np.count_nonzero(test45z)))

mask1 = np.logical_and(gt45z, test45z)

gt45x_1 = gt45x * mask1
gt45y_1 = gt45y * mask1
test45x_1 = test45x * mask1
test45y_1 = test45y * mask1

# Mean Square error in Non-ocluded areas
msen45 = np.sqrt((gt45x_1 - test45x_1) ** 2 + (gt45y_1 - test45y_1) ** 2)
msen45_r = np.reshape(msen45, [-1])[np.reshape(mask1, [-1])]

plt.figure(1)
plt.hist(msen45_r, bins=50, normed=True)
plt.title("MSE normalized histogram")
plt.ylabel("% pixels")
plt.xlabel("MSE")

m_msen45 = np.mean(np.mean(msen45_r))
print("x2mean square error (non-ocluded): " + str(m_msen45))

plt.figure(2)
plt.imshow(msen45)
plt.colorbar()
plt.title("MSE map")

mask2 = msen45 > 3.0

# Percentage of Erroneous Pixels in Non-occluded areas
pepn45 = np.count_nonzero(mask1[mask2]) / np.count_nonzero(mask1)
print("percentage of erroneous pixels (non-ocluded): " + str(pepn45) + "\n")

# Second pair
gt157_name = "../databases/data_stereo_flow/training/flow_noc/000157_10.png"
test157_name = "../test_results/motion/LKflow_000157_10.png"

gt157 = cv2.imread(gt157_name, -1)
test157 = cv2.imread(test157_name, -1)

gt157x = (np.array(gt157[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
gt157y = (np.array(gt157[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
gt157z = np.array(gt157[:, :, 0], dtype=bool)
print("non ocluded test: " + str(np.count_nonzero(gt157z)))

test157x = (np.array(test157[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
test157y = (np.array(test157[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
test157z = np.array(test157[:, :, 0], dtype=bool)
print("non ocluded test: " + str(np.count_nonzero(test157z)))

mask1 = np.logical_and(gt157z, test157z)

gt157x_1 = gt157x * mask1
gt157y_1 = gt157y * mask1
test157x_1 = test157x * mask1
test157y_1 = test157y * mask1

# Mean Square error in Non-ocluded areas
msen157 = np.sqrt((gt157x_1 - test157x_1) ** 2 + (gt157y_1 - test157y_1) ** 2)
msen157_r = np.reshape(msen157, [-1])[np.reshape(mask1, [-1])]

plt.figure(3)
plt.hist(msen157_r, bins=50, normed=True)
plt.title("MSE normalized histogram")
plt.ylabel("% pixels")
plt.xlabel("MSE")

m_msen157 = np.mean(np.mean(msen157_r))
print("x2mean square error (non-ocluded): " + str(m_msen157))

plt.figure(4)
plt.imshow(msen157)
plt.colorbar()
plt.title("MSE map")

mask2 = msen157 > 3.0

# Percentage of Erroneous Pixels in Non-occluded areas
pepn157 = np.count_nonzero(mask1[mask2]) / np.count_nonzero(mask1)
print("percentage of erroneous pixels (non-ocluded): " + str(pepn157))

plt.show()
