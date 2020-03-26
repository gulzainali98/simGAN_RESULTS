import sys
import time
import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob
from PIL import Image
import scipy.io as sio
import os
import math
from scipy import ndimage
import copy
import gc
from PIL import Image

import h5py
import gc

def store_many_hdf5(images, labels, data_type, counter, kinect_num):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    root_path= '/data/Gul Zain/'+data_type+"_dataset/"
    num_images = len(images)
    images= np.array(images)
    labels1= np.array(labels)

    print("Storing")
    print(root_path+data_type+"_"+kinect_num+"_"+counter+".h5")
    print("-------------------------")
    # Create a new HDF5 file
    file = h5py.File(root_path+data_type+"_"+kinect_num+"_"+counter+".h5", "w")

    # Create a dataset in the file
    synth_images = file.create_dataset(
        "synthetic", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    depth_images = file.create_dataset(
        "depth", np.shape(labels1), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
    del images
    del labels1
    del synth_images
    del depth_images
    gc.collect()
    print("file closed. data saved")
    print("-------------------------")



Edges = [[0, 1], [1, 2], [2, 3], [3, 4],
         [5, 6], [6, 7], [7, 8], [8, 9],
         [10, 11], [11, 12], [12, 13], [13, 14],
         [15, 16], [16, 17], [17, 18], [18, 19],
         [4, 20], [9, 21], [14, 22], [19, 23],
         [20, 24], [21, 24], [22, 24], [23, 24],
         [24, 25], [24, 26], [24, 27],
         [27, 28], [28, 29], [29, 30]]

dataset_cutoff=[14000, 2*14000,3*14000,4*14000, 5*14000]
# dataset_cutoff=[500, 2*500,3*500,4*500, 5*500]
## This part of code is modified from [DeepPrior](https://cvarlab.icg.tugraz.at/projects/hand_detection/)
def CropImage(depth, com, cube_size):
    u, v, d = com
    zstart = d - cube_size / 2.
    zend = d + cube_size / 2.

    # pricinal points are omitted (due to simplicity?)
    xstart = int(math.floor((u * d / fx - cube_size / 2.) / d * fx))
    xend = int(math.floor((u * d / fx + cube_size / 2.) / d * fx))
    ystart = int(math.floor((v * d / fy - cube_size / 2.) / d * fy))
    yend = int(math.floor((v * d / fy + cube_size / 2.) / d * fy))

    cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
    cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, depth.shape[0])),
                                (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=0)
    msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
    msk2 = np.bitwise_and(cropped > zend, cropped != 0)
    cropped[msk1] = zstart
    cropped[msk2] = zend

    dsize = (img_size, img_size)
    wb = (xend - xstart)
    hb = (yend - ystart)
    if wb > hb:
        sz = (dsize[0], (int)(hb * dsize[0] / wb))
    else:
        sz = ((int)(wb * dsize[1] / hb), dsize[1])

    roi = cropped


    rz = cv2.resize(cropped, sz)
    # maxmin = cropped.max() - cropped.min()
    # cropped_norm = (cropped - cropped.min()) / maxmin
    # rz = maxmin * resize(cropped_norm, sz, mode='reflect', preserve_range=True) + cropped.min()
    # rz = rz.astype(np.float32)

    ret = np.ones(dsize, np.float32) * zend
    xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
    xend = int(xstart + rz.shape[1])
    ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
    yend = int(ystart + rz.shape[0])
    ret[ystart:yend, xstart:xend] = rz

    return ret

def readDepth(path):
    """
    Note: In each depth png file the top 8 bits of depth are
    packed into the green channel and the lower 8 bits into blue.
    See http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#download
    Ref: [1]
    """
    rgb = Image.open(path)
    # print(rgb)
    r, g, b = rgb.split()

    r = np.asarray(r, np.int32)
    g = np.asarray(g, np.int32)
    b = np.asarray(b, np.int32)

    # dpt = b + g*256

    dpt = np.bitwise_or(np.left_shift(g, 8), b)
    imgdata = np.asarray(dpt, np.float32)
    return imgdata

##
J = 31
# joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32, 1, 2, 4, 7, 8, 10, 13, 14, 16, 19, 20, 22, 5, 11, 17, 23, 28])
joint_id = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 5, 11, 17, 23, 32, 30, 31, 28, 27, 25, 24])
img_size = 128

fx = 588.03
fy = 587.07
fu = 320.
fv = 240.

data_names = ['train', 'test_1', 'test_2']
cube_sizes = [300, 300, 300]
# id_starts = [0, 0, 2440]
# id_ends = [72756, 2440, 8252]

id_starts=[0,0,0]
id_ends=[8252,8252,8252]
#id_ends = [727, 2440, 8252]
# num_packages = [3, 1, 1]
num_packages = [1, 1, 1]
# https://github.com/jakeoung/handpose_pytorch/blob/436340e3a9aac59f04ab74c53a835ab716e769d1/code/datasets/GetH5DataNYU.py
for kinect_num in range(1,4):
    gc.collect()
    data_type='train'
    list_syn=[]
    list_depth=[]
    store_count=0
    D=0
    data_name = data_names[D]
    cube_size = cube_sizes[D]
    id_start = id_starts[D]
    id_end = id_ends[D]
    chunck_size = (id_end - id_start) / num_packages[D]

    # data_type = 'test' if data_name == 'test' else 'train'
    data_type= 'test'
    back_thresh= 169 if data_type=='train' else 150
    #data_path = '{}/{}'.format(dataset_path, data_type)  #'{}/{}'.format(dataset_path, data_type)
    data_path = '/data/Gul_Zain/nyu_hand_dataset/dataset/{}'.format(data_type)
    # data_path = '/home/jmalik/Datasets/NYU/dataset/train'
    label_path = '{}/joint_data.mat'.format(data_path)

    labels = sio.loadmat(label_path)
    # print(labels['joint_uvd'].shape)
    # break
    joint_uvd = labels['joint_uvd'][kinect_num-1]
    joint_xyz = labels['joint_xyz'][kinect_num-1]
    joint_uvd_copy = copy.deepcopy(joint_uvd)

    cnt = 0
    chunck = 0
    depth_h5, joint_h5, com_h5, inds_h5 = [], [], [], []

    counter=0
    for id in range(id_start, id_end):

#             img_path = '{}/depth_'+str(kinect_num)+'_{:07d}.png'.format(data_path, id)
#             syn_path='{}/synthdepth_'+str(kinect_num)+'_{:07d}.png'.format(data_path, id)
        img_path = '{}/depth_{:01d}_{:07d}.png'.format(data_path, kinect_num,id)
        syn_path='{}/synthdepth_{:01d}_{:07d}.png'.format(data_path, kinect_num,id)


        if not os.path.exists(img_path):
            print('{} Not Exists!'.format(img_path))
            continue
        if not os.path.exists(syn_path):
            print('{} Not Exists!'.format(syn_path))
            continue
#             print(img_path)
#             print(syn_path)
        # img = cv2.imread(img_path)
        # syn_img= cv2.imread(syn_path)
        syn_depth=readDepth(syn_path)
        depth= readDepth(img_path)

        if depth is None or syn_depth is None:
            continue

        # syn_depth= np.asarray(syn_img[:, :, 0] + syn_img[:, :, 1] * 256)
        # depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256)


#         each depth image has value between 0 and 1084

        depth = CropImage(depth, joint_uvd[id, 34], cube_size)

        com3D = joint_xyz[id, 34]
        joint = joint_xyz[id][joint_id] - com3D

        # normalize depth to [-1,1] and resize to one of the shape [128,128]
        depth = ((depth - com3D[2]) / (cube_size / 2)).reshape(1, img_size, img_size)


        syn_depth = CropImage(syn_depth, joint_uvd[id, 34], cube_size)

        com3D = joint_xyz[id, 34]
        joint = joint_xyz[id][joint_id] - com3D

        # normalize depth to [-1,1] and resize to one of the shape [128,128]
        syn_depth = ((syn_depth - com3D[2]) / (cube_size / 2)).reshape(1, img_size, img_size)
        print("synth_cropped/"+syn_path.split("/")[-1])
        # print(syn_depth[0].shape)
        cv2.imwrite("/data/Gul_Zain/my-GANs/tf_gan/tf_gan_test2/simGAN_NYU_Hand/data/hand/png_cropped_dataset/new_synt_data_cropped/"+syn_path.split("/")[-1], 255-(syn_depth[0]+1)*127.5)
