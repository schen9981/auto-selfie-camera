from preprocess import read_raw_data, train_test_split
import os
import random
import numpy as numpy
from scipy import ndimage
from sklearn.svm import SVC
import math
import matplotlib.pyplot as plt


# def train_svm():
#     # support vector classifier with radial basis kernel
#     svc = SVC(kernel='rbf')




def visualize_imgs(data_sample):
    '''
        to visualize the 400 of the 4000 images
    '''
    # initialize figures
    fig_size = 20
    num_data = fig_size ** 2
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i, data in enumerate(data_sample[0:num_data]):
        # plot the images
        p = fig.add_subplot(fig_size, fig_size, i + 1)
        p.imshow(data, cmap=plt.cm.gray)
    
    plt.show()


def main():
    '''
    main function
    '''

    # read in data and train/test split
    image_dir = '../data/genki4k/files'
    labels_dir = '../data/genki4k/labels.txt'

    data_sample, labels, headpose = read_raw_data(image_dir, labels_dir)
    visualize_imgs(data_sample)
        
    # train_data, train_labels, train_headpose, test_data, test_labels, test_headpose = train_test_split(data_sample, labels, headpose)

if __name__ == '__main__':
    main()



