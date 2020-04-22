from preprocess import read_raw_data, train_test_split
import os
import random
import numpy as numpy
from scipy import ndimage
from sklearn.svm import SVC
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import sem


def scores_cross_validation_svm(svc, data, labels, k):
    '''
        evaluates the k-fold cross validation scores: trains the model using k-1 of the folds (is subsets of our data), 
        then is validated using the remain part of the data

        svc: our support vector classifier
        k: number of splits
        data: our images to train on
        labels: labels for each image
    '''
    # initialize KFold, which divides all the samples into k groups of samples (folds) of equal sizes
    kf = KFold(n_splits=k, random_state=0, shuffle=True)

     # iterate through each of the possible splits and train
    scores = []
    for i in range(k):
        # get next split indices
        split_train_ind, split_test_ind = next(kf.split(data))

        # split data
        train_data_split = data[split_train_ind]
        train_label_split = labels[split_train_ind]

        test_data_split = data[split_test_ind]
        test_label_split = labels[split_test_ind]

        # fit data on training splits
        model = svc.fit(train_data_split, train_label_split)

        # predict on test splits and evaluate score (ie accuracy)
        predict = svc.predict(test_data_split)
        scores.append(model.score(test_data_split, test_label_split))
    
    print(scores)

def evaluate_performance(svc, train_data, train_labels, test_data, test_labels):
    '''
        evaluates performance of our svc
    '''
    # fit our svc (which was trained using kfold cross validation) to training data
    svc.fit(train_data, train_labels)

    # determine train accuracy
    print("Training Accuracy: ", svc.score(train_data, train_labels))

    # determine test accuracy
    print("Testing Accuracy", svc.score(test_data, test_labels))


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
        p.axis('off')
    
    plt.show()


def main():
    '''
    main function
    '''

    # read in data and train/test split
    image_dir = '../data/genki4k/files'
    labels_dir = '../data/genki4k/labels.txt'

    

    # read in data
    data_sample, labels, headpose = read_raw_data(image_dir, labels_dir)
    # visualize_imgs(data_sample)

    # train-test split
    train_data, train_labels, train_headpose, test_data, test_labels, test_headpose = train_test_split(data_sample, labels, headpose)

    # initialize support vector classifer with linear kernel 
    svc = SVC(kernel='linear')

    # k-fold cross-validation
    k = 5
    scores_cross_validation_svm(svc, train_data, train_labels, k)
    evaluate_performance(svc, train_data, train_labels, test_data, test_labels)
        

if __name__ == '__main__':
    main()



