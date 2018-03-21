from utils import *
import os
import numpy as np
from video_stabilization import video_stabilization

PlotsDirectory = '../plots/Week4/task2-3/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

print("reading video...")
seq_color = video_to_frame('video1.mp4', grayscale=False)

max_size = 100
seq_color = seq_color[0:max_size]

#block_size_x, block_size_y, search_area_x, search_area_y = 20, 20, 20, 20
#print("stabilizing video...")
#print(seq_color.shape)
#est_seq = video_stabilization(seq_color, block_size_x, block_size_y, search_area_x, search_area_y,
 #                             compensation='backward', grayscale=False, resize=(320, 240))

#print("saving video...")
#np.save(PlotsDirectory + 'own_stabilization.npy', est_seq)

write_images2(seq_color, PlotsDirectory, 'seq_')
