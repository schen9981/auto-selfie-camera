from preprocess import read_raw_data, train_test_split
import os
import random
import numpy as np
from scipy import ndimage
from sklearn.svm import SVC
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import sem
from skimage.feature import hog
from scipy.spatial.distance import cdist
from  sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import pickle


def flatten_data(data):
    '''
        flattens the 3d dataset of dimensions: # samples x img_size x img_size into 
        matrix of dimension: # samples x (img_size ^ 2)

        data: our 3 dimensional dataset 
        return: flattened dataset 
    '''
    num_samples = len(data)
    reshaped = data.reshape((num_samples, -1))
    return reshaped

def get_hog_features(data):
    '''
        takes input images and hog features for each input image, 
        return histograms in array (uses vocab to calculate distances)

        data: input dataset

        return: matrix of nxd, where n is the number of images in the input dataset, and d is the 
                size of the histogram built for each image 
    '''

    num_images = len(data)
    cells = 10
    pixels = 10

    hog_features = None

    for i in range(num_images):
        curr_img = data[i]

        # extract features
        if i == 0:
            hog_features = hog(curr_img, orientations=15, cells_per_block=(cells, cells), pixels_per_cell=(pixels, pixels), feature_vector=True)
        else:
            feature = hog(curr_img, orientations=15, cells_per_block=(cells, cells), pixels_per_cell=(pixels, pixels), feature_vector=True)
            hog_features = np.vstack((hog_features, feature))

    return hog_features


def scores_cross_validation_svm(model, data, labels, k, bag_of_words=False):
    '''
        evaluates the k-fold cross validation scores: trains the model using k-1 of the folds (is subsets of our data), 
        then is validated using the remain part of the data

        model: our support vector classifier
        k: number of splits
        data: our images to train on (or features)
        labels: labels for each image
        bag_of_words: boolean indicating if bag of words representaiton of images used
    '''
    # initialize KFold, which divides all the samples into k groups of samples (folds) of equal sizes
    kf = KFold(n_splits=k, random_state=0, shuffle=True)

     # iterate through each of the possible splits and train
    scores = []
    kfold_iter = kf.split(data, labels)
    for i in range(k):
        # get next split indices
        split_train_ind, split_test_ind = next(kfold_iter)

        # split data
        train_data_split = data[split_train_ind]
        train_label_split = labels[split_train_ind]

        test_data_split = data[split_test_ind]
        test_label_split = labels[split_test_ind]

        # if is not bag of words rep, flatten data
        if not bag_of_words:
            train_data_split = flatten_data(train_data_split)
            test_data_split = flatten_data(test_data_split)

        # fit data on training splits
        model.fit(train_data_split, train_label_split)

        # predict on test splits and evaluate score (ie accuracy)
        predict = model.predict(test_data_split)
        curr_score = model.score(test_data_split, test_label_split)
        scores.append(curr_score)
        print("Score on fold", i, ":", curr_score)
    
    print(scores)

def evaluate_performance(model, train_data, train_labels, test_data, test_labels, bag_of_words=False):
    '''
        evaluates performance of our model

        bag_of_words: boolean indicating if bag of words representaiton of images used
    '''
    # flatten data if not bag of words
    if not bag_of_words:
        train_data = flatten_data(train_data)
        test_data = flatten_data(test_data)

    # fit our svc (which was trained using kfold cross validation) to training data
    model.fit(train_data, train_labels)

    # determine train accuracy
    print("Training Accuracy:", model.score(train_data, train_labels))

    # determine test accuracy
    print("Testing Accuracy:", model.score(test_data, test_labels))



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
    # read in data and train/test split
    labels_dir = '../data/genki4k/labels.txt'
    cropped_dir = '../data/genki4k/files/cropped'

    # read in data
    data_sample, labels, headpose = read_raw_data(cropped_dir, labels_dir)
    # visualize_imgs(data_sample)

    # train-test split
    train_data, train_labels, train_headpose, test_data, test_labels, test_headpose = train_test_split(data_sample, labels, headpose)
    
    # get hog features
    train_features = get_hog_features(train_data)
    test_features = get_hog_features(test_data)
    # print(test_features.shape)
    print("got features")

    # feed features into pca 
    pca = PCA(n_components=0.90)
    pca.fit(train_features)
    train_features = pca.transform(train_features)
    test_features = pca.transform(test_features)

    # initialize support vector classifer with linear kernel 
    svc = SVC(kernel='linear', probability=False, C=5)

    # k-fold cross-validation w/o bag_of_words
    # k = 5
    # scores_cross_validation_svm(svc, train_data, train_labels, k)
    # evaluate_performance(svc, train_data, train_labels, test_data, test_labels)

    # k-fold cross validation w/ bag_of_words
    k = 5
    scores_cross_validation_svm(svc, train_features, train_labels, k, bag_of_words=True)
    evaluate_performance(svc, train_features, train_labels, test_features, test_labels, bag_of_words=True)

    # saving trained model
    filename = 'trained_model.sav'
    pickle.dump(svc, open(filename, 'wb'))

    # saving pca transformation
    pca_filename = 'trained_feature_transform.sav'
    pickle.dump(pca, open(pca_filename, 'wb'))

        

if __name__ == '__main__':
    main()



