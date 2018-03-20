import numpy as np
import cv2


# give us the center of a block/region
def get_block_center(block):
    center_x = int(block.shape[0]/2)
    center_y = int(block.shape[1]/2)
    return np.array([center_x, center_y])

# get the movement between 2 points, usefull to compute the movement of the block's center.
def get_motion(point1, point2):
    motion_x = point2[0] - point1[0]
    motion_y = point2[1] - point1[1]
    return  np.array([motion_x, motion_y])

def get_MSD(block1, block2):
    """
     Mean Square Difference between block1 and block2.
     The MSD squares the difference between pixels. This exaggerates any differences.
    """
    #print(block1.shape)
    #print(block2.shape)
    return sum(sum(abs(block1 - block2) ** 2))

def get_matching_in_search_area(block, region):
    """
    Search a "block" in a given "region".
    """

    region_size_x, region_size_y = region.shape     #[:2]

    # get the center of the region, which is also the center of the block to search in the region.
    # point1 is the center of the block referenced inside the relative positions of the region.
    point1 = get_block_center(region)

    min_diff = float("inf")  # assign "positive infinite"

    block_size_x, block_size_y = block.shape

    # One block is searched in a search area ("region"), and will produce only on motion in x and y directions.
    motion_xy = np.zeros([2], dtype=np.uint8)

    for row in range(region_size_x - block_size_x + 1 ):
        for column in range(region_size_y - block_size_y + 1):
            block2compair = region[row : row + block_size_x, column : column+block_size_y]
            diff = get_MSD(block, block2compair)    # We use as matching criteria: Mean Square Difference
            if diff < min_diff:
                min_diff = diff

                # compute the location inside the region
                center2 = get_block_center(block2compair)
                point2 = np.array([ row + center2[0], column + center2[1] ])
                motion_xy = get_motion(point1, point2)

                # dedicated to the lovers of write all just in one line of code!
                #motion_xy = get_motion( get_block_center(region), np.array([ row + get_block_center(block2compair)[0], column + get_block_center(block2compair)[1] ]) )

    return min_diff, motion_xy

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
        sa_x2 = row_limit

    if sa_y1 < 0:
        sa_y1 = 0
    if sa_y2 > column_limit:
        sa_y2 = column_limit


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

    motion = np.zeros([n_blocks_x, n_blocks_y, 2]) # 2 dimension because we will have the motion in x and y.

    # add padding to the "search_img" to compute the last border ????

    for row in range(n_blocks_x):
        for column in range (n_blocks_y):
            #block = reference_img[row*n_blocks_x:row*n_blocks_x+n_blocks_x, column*n_blocks_y:column*n_blocks_y+n_blocks_y]

            x1, x2 = row*n_blocks_x, row*n_blocks_x+n_blocks_x
            y1, y2 = column*n_blocks_y, column*n_blocks_y+n_blocks_y

            block = reference_img[x1:x2, y1:y2]

            #sa_x1, sa_y1, sa_x2, sa_y2 = get_area_search(reference_img, x1, y1, x2, y2, search_area_x, search_area_y)

            #_, block_motion_xy = get_matching_in_search_area(block, region)

            #motion[row, column, 0] = block_motion_xy[0]
            #motion[row, column, 1] = block_motion_xy[1]

    return motion

#====================> TESTING <=============================

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

# CASE #1: find area search for a block located in the upper left of the image
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