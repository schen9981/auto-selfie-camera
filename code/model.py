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

def build_vocabulary(data, vocab_size):
    '''
        samples hog desciptors from input training images, cluster with kmeans,
        return cluster centers

        data: raw image pixel values 
        vocab_size: integer indicating the number of words desired for 
                    bag of words vocab set
    '''
    num_images = len(data)
    cells = 4
    pixels = 8

    # get all features of images
    hog_descriptors = None

    for i in range(num_images):
        curr_img = data[i]

        # generate hog descriptor for image
        hog_img = np.array(hog(curr_img, cells_per_block=(cells, cells), pixels_per_cell=(pixels, pixels), feature_vector=True))

        # reshape into list of block feature vectors
        if i == 0:
            hog_descriptors = np.reshape(hog_img, (-1, cells * cells * 9))
        else:
            hog_img = np.reshape(hog_img, (-1, cells * cells * 9))
            hog_descriptors = np.concatenate((hog_descriptors, hog_img))
    
    # kmeans clustering to find smaller number of representation points
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=300, max_iter=100, tol=0.001).fit(hog_descriptors)

    return kmeans.cluster_centers_

def get_bag_of_words(data):
    '''
        takes input images and calculates the bag of words histogram for each input image, 
        return histograms in array (uses vocab to calculate distances)

        data: input dataset

        return: matrix of nxd, where n is the number of images in the input dataset, and d is the 
                size of the histogram built for each image 
    '''
    vocab = np.load('vocab.npy')
    print("vocab loaded")

    num_images = len(data)
    vocab_size = len(vocab)
    cells = 4
    pixels = 8

    bag_of_words = np.zeros((num_images, vocab_size))

    for i in range(num_images):
        curr_img = data[i]

        # extract features
        hog_img = hog(curr_img, cells_per_block=(cells, cells), pixels_per_cell=(pixels, pixels), feature_vector=True)

        # reshape into list of block feature vectors
        hog_img = np.reshape(hog_img, (-1, cells * cells * 9))

        # get distances
        distances = cdist(hog_img, vocab, 'cosine')

        for k in range(len(hog_img)):
            curr_dists = distances[k]
            sorted_ind = np.argsort(curr_dists)
            bag_of_words[i, sorted_ind[0]] += 1
        
        # normalize histogram
        bag_of_words[i, :] = bag_of_words[i, :] / np.linalg.norm(bag_of_words[i, :])

        return bag_of_words


def scores_cross_validation_svm(svc, data, labels, k, bag_of_words=False):
    '''
        evaluates the k-fold cross validation scores: trains the model using k-1 of the folds (is subsets of our data), 
        then is validated using the remain part of the data

        svc: our support vector classifier
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
        model = svc.fit(train_data_split, train_label_split)

        # predict on test splits and evaluate score (ie accuracy)
        predict = svc.predict(test_data_split)
        curr_score = model.score(test_data_split, test_label_split)
        scores.append(curr_score)
        print("Score on fold", i, ":", curr_score)
    
    print(scores)

def evaluate_performance(svc, train_data, train_labels, test_data, test_labels, bag_of_words=False):
    '''
        evaluates performance of our svc

        bag_of_words: boolean indicating if bag of words representaiton of images used
    '''
    # flatten data if not bag of words
    if not bag_of_words:
        train_data = flatten_data(train_data)
        test_data = flatten_data(test_data)

    # fit our svc (which was trained using kfold cross validation) to training data
    svc.fit(train_data, train_labels)

    # determine train accuracy
    print("Training Accuracy:", svc.score(train_data, train_labels))

    # determine test accuracy
    print("Testing Accuracy:", svc.score(test_data, test_labels))


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
    image_dir = '../data/genki4k/files'
    labels_dir = '../data/genki4k/labels.txt'

    # read in data
    data_sample, labels, headpose = read_raw_data(image_dir, labels_dir)
    # visualize_imgs(data_sample)

    # train-test split
    train_data, train_labels, train_headpose, test_data, test_labels, test_headpose = train_test_split(data_sample, labels, headpose)

    # build vocabulary from training data
    if not os.path.isfile('vocab.npy'):
        print("no existing vocab file, generating from training data")
        vocab_size = 10
        vocab = build_vocabulary(train_data, vocab_size)
        # save vocab file
        np.save('vocab.npy', vocab)
    
    # get bag of words 
    train_features = get_bag_of_words(train_data)
    test_features = get_bag_of_words(test_data)

    # initialize support vector classifer with linear kernel 
    svc = SVC(kernel='rbf')

    # k-fold cross-validation w/o bag_of_words
    # k = 5
    # scores_cross_validation_svm(svc, train_data, train_labels, k)
    # evaluate_performance(svc, train_data, train_labels, test_data, test_labels)

    # k-fold cross validation w/ bag_of_words
    k = 5
    scores_cross_validation_svm(svc, train_features, train_labels, k, bag_of_words=True)
    evaluate_performance(svc, train_features, train_labels, test_features, test_labels, bag_of_words=True)

        

if __name__ == '__main__':
    main()



