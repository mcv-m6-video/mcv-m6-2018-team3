import numpy as np
import cv2
import time


# give us the center of a block/region
# def get_block_center(block):
#     center_x = int(block.shape[0]/2)
#     center_y = int(block.shape[1]/2)
#     return np.array([center_x, center_y])
#
# # get the movement between 2 points, usefull to compute the movement of the block's center.
# def get_motion(point1, point2):
#     motion_x = point2[0] - point1[0]
#     motion_y = point2[1] - point1[1]
#     return  np.array([motion_x, motion_y])

def get_MSD(block1, block2):
    """
     Mean Square Difference between block1 and block2.
     The MSD squares the difference between pixels. This exaggerates any differences.
    """
    #print(block1.shape)
    #print(block2.shape)
    return sum(sum(abs(block1 - block2) ** 2))

def get_matching_in_search_area(block, search_img, block_coord, search_coord, thresh=None):

    #exact indexation!
    #block_coord = (x1, x2, y1, y2)
    #search_coord = (x1, x2, y1, y2)

    #return motion_x, motion_y, valid

    px_top = abs(search_coord[0] - block_coord[0])
    px_down = abs(search_coord[1] - block_coord[1])
    px_left = abs(search_coord[2] - block_coord[2])
    px_right = abs(search_coord[3] - block_coord[3])

    #print("matching")
    #print(px_top, px_down, px_left, px_right)

    motion_map = np.zeros([px_top + px_down + 1, px_left + px_right + 1])
    #print(motion_map.shape)

    x_range = np.arange(-px_top, (px_down + 1))
    y_range = np.arange(-px_left, (px_right + 1))

    for idx, mx in enumerate(x_range):
        for idy, my in enumerate(y_range):
            #print(block.shape)
            #print(search_img[block_coord[0]+mx:block_coord[1]+mx +1,
            #                                      block_coord[2] + my:block_coord[3] + my +1].shape)

            motion_map[idx, idy] = get_MSD(block, search_img[block_coord[0]+mx:block_coord[1]+mx +1,
                                                  block_coord[2] + my:block_coord[3] + my +1])


    if thresh is not None:
        motion_map[motion_map<thresh] = float('inf')

    arg_min = np.where(motion_map == np.min(motion_map))

    if np.min(motion_map) == float('inf'):
        return 0, 0, 0

    else:
        mx = x_range[arg_min[0]][0]
        my = y_range[arg_min[1]][0]
        return motion_map, mx, my, 1

# OUTPU: coordinates of the area search, upper left and bottom right points.
def get_area_search(reference_img, x1, x2, y1, y2, search_area_x, search_area_y):

    sa_x1 = x1 - search_area_x
    sa_x2 = x2 + search_area_x

    sa_y1 = y1 - search_area_y
    sa_y2 = y2 + search_area_y

    # getting the limits of the image
    (row_limit, column_limit) = reference_img.shape[:2]

    # The rows coordinate of the search area MUST be inside a valid area in the image
    if sa_x1 < 0:
        sa_x1 = 0
    if sa_x2 > row_limit:
        sa_x2 = row_limit-1

    if sa_y1 < 0:
        sa_y1 = 0
    if sa_y2 > column_limit:
        sa_y2 = column_limit-1


    return sa_x1, sa_x2, sa_y1, sa_y2

def get_block_matching(curr_img, prev_img, block_size_x, block_size_y, search_area_x, search_area_y, compensation='backward'):

    if compensation == 'backward':
        reference_img = curr_img
        search_img = prev_img
    else:
        reference_img = prev_img
        search_img = curr_img

    n_blocks_x = int(reference_img.shape[0] / block_size_x)
    n_blocks_y = int(reference_img.shape[1] / block_size_y)

    #print(reference_img.shape)
    #print(n_blocks_x, n_blocks_y)

    motion = np.zeros(curr_img.shape+(3,))  # motion has the same size of the image and 3 channels.

    for row in range(n_blocks_x):
        for column in range (n_blocks_y):
            #block = reference_img[row*n_blocks_x:row*n_blocks_x+n_blocks_x, column*n_blocks_y:column*n_blocks_y+n_blocks_y]
            #print('\n')
            #print(row, column)
            x1, x2 = row*block_size_x, row*block_size_x+block_size_x - 1
            y1, y2 = column*block_size_y, column*block_size_y+block_size_y - 1
            #print(x1, x2, y1, y2)
            block = reference_img[x1:x2+1, y1:y2+1]
            #print(block.shape)

            sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(reference_img, x1, x2, y1, y2, search_area_x, search_area_y)
            #print(sa_x1, sa_x2, sa_y1, sa_y2)
            # exact indexation!
            # block_coord = (x1, x2, y1, y2)
            # search_coord = (x1, x2, y1, y2)

            motion_x, motion_y, valid = get_matching_in_search_area(block, search_img, (x1, x2, y1, y2), (sa_x1, sa_x2, sa_y1, sa_y2) , thresh=None)

            motion[x1:x2+1, y1:y2+1, 0] = np.ones(block.shape)*motion_x
            motion[x1:x2+1, y1:y2+1, 1] = np.ones(block.shape) * motion_y
            motion[x1:x2+1, y1:y2+1, 2] = np.ones(block.shape) * valid


    return motion

#====================> TESTING <=============================

"""
t1 = time.time()
curr_img = cv2.imread("../../databases/traffic/input/in000950.jpg",0)
prev_img = cv2.imread("../../databases/traffic/input/in000951.jpg",0)

motion = get_block_matching(curr_img, prev_img, 20, 20, 10 , 10)

print(time.time() - t1)
"""




#block_test = np.zeros([3,3])
#center = get_block_center(block_test)

#reference_img = cv2.imread("../../databases/traffic/input/in000950.jpg",0)
#current_img = cv2.imread("../../databases/traffic/input/in000991.jpg",0)

#block = current_img[3:7,3:7]
#region = reference_img[10:100,10:100]
#motion = get_matching(block, region)
#motion = get_block_matching(current_img, reference_img, 3, 3, 2, 2, compensation='backward')

#opticalFlow_img = compute_block_matching(reference_img[], current_img[])


# ===> test2 <===

# 5x5
# curr_img= np.array([[ 40, 109, 117, 233,  72],
#                     [108, 238, 120, 184,  16],
#                     [ 87, 194,  41,  32, 255],
#                     [208,  28,  74, 239, 121],
#                     [129, 250, 145,  10, 212]])
#
# block = curr_img[0:2,0:2]  #np.random.randint(0, 255, size=(3,3))
#
#
# prev_img= np.array([[ 50, 119, 127, 233,  72],
#                     [118, 248, 130, 184,  16],
#                     [ 97, 204,  41, 110, 118],
#                     [208,  28, 109, 239, 121],
#                     [129, 250,  88, 195,  42]])
#
# region = prev_img
# min_diff, motion_xy = get_matching_in_search_area(block, region)
# print(min_diff)
# print(motion_xy)


# ===> TEST #3: get_area_search <===

# # CASE #1: find area search for a block located in the upper left of the image
# curr_img = np.random.randint(0, 255, size=(7,7))
# print("current image:")
# print(curr_img)
# print("\n")
#
# x1, x2, y1, y2 = 0, 2, 0, 2
# block = curr_img[x1:x2,y1:y2]
# print("block:")
# print (block)
# print("\n")
#
# search_area_x = 1
# search_area_y = 1
#
# sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(curr_img, x1, x2, y1, y2, search_area_x, search_area_y)
#
# print("coordinates of search area:")
# print(sa_x1, sa_x2, sa_y1, sa_y2)
# print("\n")
#
# region = curr_img[sa_x1:sa_x2, sa_y1:sa_y2]
# print("region:")
# print(region)
# print("\n")


# CASE #2: find area search for a block located in the upper right of the image
# curr_img = np.random.randint(0, 255, size=(7,7))
# print("current image:")
# print(curr_img)
# print("\n")
#
# x1, x2, y1, y2 = 0, 3, 4, 7
# block = curr_img[x1:x2,y1:y2]
# print("block:")
# print (block)
# print("\n")
#
# search_area_x = 2
# search_area_y = 2
#
# sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(curr_img, x1, x2, y1, y2, search_area_x, search_area_y)
#
# print("coordinates of search area:")
# print(sa_x1, sa_x2, sa_y1, sa_y2)
# print("\n")
#
# region = curr_img[sa_x1:sa_x2, sa_y1:sa_y2]
# print("region:")
# print(region)
# print("\n")


# CASE #3: find area search for a block located in the bottom left of the image
# curr_img = np.random.randint(0, 255, size=(7,7))
# print("current image:")
# print(curr_img)
# print("\n")
#
# x1, x2, y1, y2 = 4, 7, 0, 3
# block = curr_img[x1:x2,y1:y2]
# print("block:")
# print (block)
# print("\n")
#
# search_area_x = 3
# search_area_y = 3
#
# sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(curr_img, x1, x2, y1, y2, search_area_x, search_area_y)
#
# print("coordinates of search area:")
# print(sa_x1, sa_x2, sa_y1, sa_y2)
# print("\n")
#
# region = curr_img[sa_x1:sa_x2, sa_y1:sa_y2]
# print("region:")
# print(region)
# print("\n")




# CASE #4: find area search for a block located in the bottom right of the image
# curr_img = np.random.randint(0, 255, size=(7,7))
# print("current image:")
# print(curr_img)
# print("\n")
#
# x1, x2, y1, y2 = 4, 7, 4, 7
# block = curr_img[x1:x2,y1:y2]
# print("block:")
# print (block)
# print("\n")
#
# search_area_x = 3
# search_area_y = 3
#
# sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(curr_img, x1, x2, y1, y2, search_area_x, search_area_y)
#
# print("coordinates of search area:")
# print(sa_x1, sa_x2, sa_y1, sa_y2)
# print("\n")
#
# region = curr_img[sa_x1:sa_x2, sa_y1:sa_y2]
# print("region:")
# print(region)
# print("\n")

# CASE #5: find area search for a block located in the middle of the image
# curr_img = np.random.randint(0, 255, size=(7,7))
# print("current image:")
# print(curr_img)
# print("\n")
#
# x1, x2, y1, y2 = 3, 5, 3, 5
# block = curr_img[x1:x2,y1:y2]
# print("block:")
# print (block)
# print("\n")
#
# search_area_x = 4
# search_area_y = 4
#
# sa_x1, sa_x2, sa_y1, sa_y2 = get_area_search(curr_img, x1, x2, y1, y2, search_area_x, search_area_y)
#
# print("coordinates of search area:")
# print(sa_x1, sa_x2, sa_y1, sa_y2)
# print("\n")
#
# region = curr_img[sa_x1:sa_x2, sa_y1:sa_y2]
# print("region:")
# print(region)
# print("\n")