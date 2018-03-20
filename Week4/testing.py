import cv2
import numpy as np

# First pair
gt45_name = "../../databases/data_stereo_flow/training/flow_noc/000045_10.png"
test45_name1 = "../../databases/data_stereo_flow/training/image_0/000045_10.png"
test45_name2 = "../../databases/data_stereo_flow/training/image_0/000045_11.png"
names = [gt45_name, test45_name1, test45_name2]
gt = cv2.imread(names[0], -1)
prvs = cv2.imread(names[1], 0)
act = cv2.imread(names[2], 0)

flow = cv2.calcOpticalFlowFarneback(prvs, act, None, 0.5, 3, 15, 3, 5, 1.2, 0)
print(type(flow[0,0,0]))
print(max(flow.reshape(-1)))
print(min(flow.reshape(-1)))
