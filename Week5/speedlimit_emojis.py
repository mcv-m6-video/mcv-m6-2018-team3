from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt

def RGBtoBGR(img):
    temp = img[:,:,0].copy()
    img[:,:,0] = img[:,:,2].copy()
    img[:,:,2] = temp
    return img

def speedlimit_emojis(sequence, image, speed, bb_pos, bb_shape):
    #-------------------CONFIGURATION------------------------
    #Highway
    sig = 50 #signal shape

    #Load emojis
    e1 = ndimage.imread("./emojis/1.png",mode='RGB')
    e2 = ndimage.imread("./emojis/2.png",mode='RGB')
    e3 = ndimage.imread("./emojis/3.png",mode='RGB')
    s70 = ndimage.imread("./emojis/s70.png", mode='RGB')

    e1 = misc.imresize(e1, (round(float(bb_shape[1]) * 0.2), round(float(bb_shape[1]) * 0.2), 3))
    e2 = misc.imresize(e2, (round(float(bb_shape[1])*0.2), round(float(bb_shape[1])*0.2), 3))
    e3 = misc.imresize(e3, (round(float(bb_shape[1]) * 0.2), round(float(bb_shape[1]) * 0.2), 3))
    s70 = misc.imresize(s70, (sig, sig, 3))

    e1 = RGBtoBGR(e1)
    e2 = RGBtoBGR(e2)
    e3 = RGBtoBGR(e3)
    s70 = RGBtoBGR(s70)

    # plt.figure()
    # plt.imshow(e2)
    # plt.show()

    if speed > 70:
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        print(bb_pos)
        bb_pos = (bb_pos[0], bb_pos[1] + 5)
        print(bb_pos)
        print(e2.shape)
        print(image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape)
        s1 = image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape
        s2 = e2.shape
        print('\n\n\n\n\n\n\n\n\n\n\n\n')

        if s1 == s2:
            image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0], :] = e2

    else:
        bb_pos = (bb_pos[0], bb_pos[1] + 5)
        s1 = image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape
        s2 = e2.shape

        if s1 == s2:
            image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0], :] = e1

    imsh = image.shape
    print(imsh)
    image[imsh[0]-sig-30 : imsh[0]-30, imsh[1]-sig-10: imsh[1]-10, :] = s70

    return image

