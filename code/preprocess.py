import os
import random
import numpy as np
from scipy import ndimage
from skimage import color
import matplotlib.pyplot as plt
import cv2

num_classes = 2
train_ratio = 0.8

img_size = 128

def read_raw_data(img_path, data_path):
    '''
        reads in raw data and labels to be used for training and testing.

        img_path: path to raw jgp images of faces
        data_path: path to the labels file, with head-pose information

        return: data_sample - # files x image_size x image_size
                labels - # files x 1
                headpose - # files x 3 (yaw, pitch, roll)
    '''
    # get list of image files
    img_files = os.listdir(img_path)

    # allocate space in memory for images
    data_sample = np.ndarray(shape=(len(img_files), img_size, img_size), dtype=np.float32)

    # allocate space in memory for corresponding lables and face pose information
    labels = np.zeros(len(img_files))
    headpose = np.zeros((len(img_files), 3))

    # store image data 
    for i, img in enumerate(img_files):
        # read image
        curr_path = os.path.join(img_path, img)
        curr_img = cv2.imread(curr_path).astype('float32')
        # convert to grayscale
        grey_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # normalize image
        normalized = np.zeros(grey_img.shape)
        normalized = cv2.normalize(grey_img, normalized, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # resize image 
        save = cv2.resize(normalized, (img_size, img_size))
        data_sample[i, :, :] = save


    # store labels and headpose
    label_file = open(data_path, "r")
    ind = 0
    for l in label_file:
        l_contents = l.split()
        labels[ind] = l_contents[0]
        headpose[ind] = l_contents[1:]
        ind += 1
    label_file.close()

    return data_sample, labels, headpose


def train_test_split(data_sample, labels, headpose):
    '''
        splits data using train-test split ration specified with train_ratio

        return: training images, training labels, training headpose data
                testing images, testing lables, testing headpose data
    '''
    # shuffle indices
    num_data = len(data_sample)
    inds = [i for i in range(num_data)]
    np.random.shuffle(inds)

    data_sample = data_sample[inds]
    labels = labels[inds]
    headpose = headpose[inds]

    # create train and test dataset 
    train_ind_end = int(train_ratio * num_data)
    test_ind_start = train_ind_end + 1

    train_data = data_sample[0:train_ind_end]
    train_labels = labels[0:train_ind_end]
    train_headpose = headpose[0:train_ind_end]

    test_data = data_sample[test_ind_start:]
    test_labels = labels[test_ind_start:]
    test_headpose = headpose[test_ind_start:]

    return train_data, train_labels, train_headpose, test_data, test_labels, test_headpose


        


    



    
    